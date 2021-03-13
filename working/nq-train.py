#%% md

### Importing Libraries

#%%

import os
import sys

RUN_ON = 'kaggle' if os.path.exists('/kaggle') else 'gcp'

if RUN_ON == 'gcp':
    os.chdir('/home/jupyter/kaggle/working')
    sys.path.extend(['../input/bert-joint-baseline/'])

#%%

import gzip
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

import bert_utils
import modeling
import tokenization
import bert_optimization as optimization

from tqdm.auto import tqdm
import importlib

importlib.reload(bert_utils)
K.clear_session()

tf.__version__


#%% md

### Classes & Functions

#%%

class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                 **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `TDense` layer with "
                            "non-floating point (and non-complex) "
                            "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to "
                             "`TDense` should be defined. "
                             "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size, last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.kernel, transpose_b=True) + self.bias


class Squeeze(tf.keras.layers.Layer):
    def call(self, x, axis=None, name=None):
        return tf.squeeze(x, axis, name)


def mk_model(config, is_training=False):
    if not is_training:
        config['hidden_dropout_prob'] = 0.0
        config['attention_probs_dropout_prob'] = 0.0
    seq_len = config['max_position_embeddings']

    #     unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')
    BERT = modeling.BertModel(config=config, name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)

    logits = TDense(2, name='logits')(sequence_output)
    start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = Squeeze(name='start_logits_or_probs')(start_logits, axis=-1)
    end_logits = Squeeze(name='end_logits_or_probs')(end_logits, axis=-1)

    ans_type = TDense(5, name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [input_ids, input_mask, segment_ids]
                           if input_ is not None],
                          [start_logits, end_logits, ans_type],
                          name='bert-baseline')


# nq loss function
def crossentropy_from_logits(y_true, y_pred):
    one_hot_positions = y_true
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    loss = -tf.reduce_mean(
        tf.reduce_sum(one_hot_positions * log_probs, axis=-1))

    return loss


#%%

class DummyObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def url_exists(url):
    """test local or gs file exists or not."""
    from urllib import parse
    res = parse.urlparse(url)
    if res.scheme == 'gs':
        # blob_name has no '/' prefix
        bucket_name, blob_name = res.netloc, res.path[1:]
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    else:
        return os.path.exists(res.path)


def make_decoder(seq_length, is_training):
    if is_training:
        feature_description = {
            "unique_ids": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "start_positions": tf.io.FixedLenFeature([], tf.int64),
            "end_positions": tf.io.FixedLenFeature([], tf.int64),
            "answer_types": tf.io.FixedLenFeature([], tf.int64)
        }
    else:
        feature_description = {
            "unique_id": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        }

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(serialized=record, features=feature_description)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for key in [k for k in example.keys() if k not in ['example_id', 'unique_ids']]:
            example[key] = tf.cast(example[key], dtype=tf.int32)
        if is_training:
            features = {
                'input_ids': example['input_ids'],
                'input_mask': example['input_mask'],
                'segment_ids': example['segment_ids']
            }
            labels = {
                'start_logits_or_probs': tf.one_hot(example['start_positions'],
                                                    depth=seq_length, dtype=tf.float32),
                'end_logits_or_probs': tf.one_hot(example['end_positions'],
                                                  depth=seq_length, dtype=tf.float32),
                'ans_type': tf.one_hot(example['answer_types'],
                                       depth=len(ANSWER_TYPE_ORDER), dtype=tf.float32)
            }
            return (features, labels)
        else:
            return example

    return _decode_record

#%%

# Configurations
FLAGS = DummyObject(skip_nested_contexts=True,
                    max_position=50,
                    max_contexts=48,
                    max_query_length=64,
                    max_seq_length=512,
                    doc_stride=128,
                    include_unknowns=-1.0,
                    n_best_size=20,
                    max_answer_length=30,
                    batch_size=64,
                    is_training=True,
                    #                     train_num_precomputed=494670,
                    train_num_precomputed=200 * 64,
                    learning_rate=3e-5,
                    num_train_epochs=3,
                    )

if RUN_ON == 'gcp':
    INPUT_PATH = 'gs://tyu-kaggle/input/'
else:
    INPUT_PATH = '../input/'
BERT_CONFIG_PATH = os.path.join('../input', 'bert-joint-baseline/bert_config.json')
CKPT_PATH = os.path.join(INPUT_PATH, 'bert-joint-baseline/model_cpkt-1')
MODEL_SAVE_PATH = './bert_trained/weights.h5'
VOCAB_PATH = os.path.join(INPUT_PATH, 'bert-joint-baseline/vocab-nq.txt')

NQ_TEST_JSONL_PATH = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'
NQ_TRAIN_JSONL_PATH = '../input/tensorflow2-question-answering/simplified-nq-train.jsonl'

NQ_TEST_TFRECORD_PATH = './nq-test.tfrecords'
NQ_TRAIN_TFRECORD_PATH = os.path.join(INPUT_PATH, 'bert-joint-baseline/nq-train.tfrecords')

SAMPLE_SUBMISSION_PATH = '../input/tensorflow2-question-answering/sample_submission.csv'

TEST_DS_TYPE = 'public' if os.path.getsize(NQ_TEST_JSONL_PATH) < 20000000 else 'private'

ANSWER_TYPE_ORDER = ['UNKNOWN', 'YES', 'NO', 'SHORT', 'LONG']

with open(BERT_CONFIG_PATH, 'r') as f:
    config = json.load(f)
print(json.dumps(config, indent=4))

n_train_instances = FLAGS.train_num_precomputed
n_total_train_steps = int(FLAGS.num_train_epochs * n_train_instances / FLAGS.batch_size)
n_epochs = FLAGS.num_train_epochs
n_steps_per_epoch = n_train_instances // FLAGS.batch_size

optimizer = optimization.create_optimizer(FLAGS.learning_rate,
                                          n_total_train_steps,
                                          num_warmup_steps=None)

#%%

# Detect hardware, return appropriate distribution strategy
try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', TPU.cluster_spec().as_dict()['worker'])
except ValueError:
    TPU = None

if TPU:
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

#%%


#%%

with strategy.scope():
    model = mk_model(config, is_training=True)
    model.summary()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(CKPT_PATH).assert_consumed()

#%%

raw_ds = tf.data.TFRecordDataset(NQ_TRAIN_TFRECORD_PATH)
if FLAGS.is_training:
    raw_ds = raw_ds.repeat()
    raw_ds = raw_ds.shuffle(buffer_size=100)
decoded_ds = raw_ds.map(make_decoder(seq_length=FLAGS.max_seq_length, is_training=FLAGS.is_training))
batched_ds = decoded_ds.batch(batch_size=FLAGS.batch_size, drop_remainder=(TPU is not None))

model.compile(optimizer,
              loss={
                  'start_logits_or_probs': crossentropy_from_logits,
                  'end_logits_or_probs': crossentropy_from_logits,
                  'ans_type': crossentropy_from_logits},
              loss_weights={
                  'start_logits_or_probs': 1,
                  'end_logits_or_probs': 1,
                  'ans_type': 1})
hist = model.fit(batched_ds, steps_per_epoch=n_steps_per_epoch, epochs=n_epochs, verbose=1)

#%%

# raw_ds = tf.data.TFRecordDataset(NQ_TFRECORD_PATH)
# decoded_ds = raw_ds.map(make_decoder(seq_length=F.max_seq_length, is_training=F.is_training))
# batched_ds = decoded_ds.batch(batch_size=32, drop_remainder=(TPU is not None))
# batched_ds.element_spec
# for x, y in batched_ds:
#     print(y)
#     break
model.save_weights(MODEL_SAVE_PATH)


# model.get_weights()


#%%


#%%


