#%% md

### Importing Libraries

#%%

import os
import sys

import json
import tensorflow as tf

FLAGS = {
   'skip_nested_contexts': True,
   'max_position': 50,
   'max_contexts': 48,
   'max_query_length': 64,
   'max_seq_length': 512,
   'doc_stride': 128,
   'include_unknowns': -1.0,
   'n_best_size': 20,
   'max_answer_length': 30,
}
F = FLAGS

RUN_ON = 'kaggle' if os.path.exists('/kaggle') else 'gcp'
# #### Configurations. change by user.
if RUN_ON == 'gcp':
    os.chdir('/home/jupyter/kaggle/working')
    sys.path.extend(['../input/bert-joint-baseline/'])
    INPUT_PATH = 'gs://tyu-kaggle/input/'
    MODEL_LOAD_DIR = './bert_trained/'
else:
    INPUT_PATH = '../input/'
    MODEL_LOAD_DIR = '../input/berttrained3/'

# #### Configurations. No need to change.
BERT_CONFIG_PATH = os.path.join(INPUT_PATH, 'bert-joint-baseline/bert_config.json')
VOCAB_PATH = os.path.join(INPUT_PATH, 'bert-joint-baseline/vocab-nq.txt')

NQ_TEST_JSONL_PATH = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'
NQ_TRAIN_JSONL_PATH = '../input/tensorflow2-question-answering/simplified-nq-train.jsonl'
SAMPLE_SUBMISSION_PATH = '../input/tensorflow2-question-answering/sample_submission.csv'
# we pad example_id, so we must redo preprocess...
NQ_TEST_TFRECORD_PATH = 'nq-test.tfrecords'

MODEL_LOAD_PATH = MODEL_LOAD_DIR + 'weights.h5'

TEST_DS_TYPE = 'public' if os.path.getsize(NQ_TEST_JSONL_PATH) < 20000000 else 'private'

SEQ_LENGTH = F['max_seq_length']  # bert_config['max_position_embeddings']

ANSWER_TYPE_ORDER = ['UNKNOWN', 'YES', 'NO', 'SHORT', 'LONG']

with tf.io.gfile.GFile(BERT_CONFIG_PATH, 'r') as f:
    bert_config = json.load(f)
print(json.dumps(bert_config, indent=4))

!ls -lh $MODEL_LOAD_DIR
!cat $MODEL_LOAD_DIR/fingerprint.json

#%%

import gzip

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

import bert_utils
import modeling
import tokenization

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


def mk_model(config):
    seq_len = config['max_position_embeddings']
    unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')
    BERT = modeling.BertModel(config=config, name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)
    logits = TDense(2, name='logits')(sequence_output)
    start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = Squeeze(name='start_logits')(start_logits, axis=-1)
    end_logits = Squeeze(name='end_logits')(end_logits, axis=-1)

    ans_type = TDense(5, name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [unique_id, input_ids, input_mask, segment_ids]
                           if input_ is not None],
                          [unique_id, start_logits, end_logits, ans_type],
                          name='bert-baseline')

#%%



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
    BATCH_SIZE = 64
    # drop_remainder must be True if running on TPU, maybe a bug
    # so we pad some examples.
    nq_test_jsonl_path2 = NQ_TEST_JSONL_PATH + '.pad'
    !cp $NQ_TEST_JSONL_PATH $nq_test_jsonl_path2
    !tail -n 3 $NQ_TEST_JSONL_PATH >> $nq_test_jsonl_path2
    NQ_TEST_JSONL_PATH = nq_test_jsonl_path2
else:
    strategy = tf.distribute.get_strategy()
    BATCH_SIZE = 16

print("REPLICAS: ", strategy.num_replicas_in_sync)

#%%

with strategy.scope():
    model = mk_model(bert_config)
    model.summary()
    model.load_weights(MODEL_LOAD_PATH)

#%%

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
        blob = bucket.blob(blob_name[1:])
        return blob.exists()
    else:
        return os.path.exists(res.path)


FEATURE_DESCRIPTION = {
    #     "example_id": tf.io.FixedLenFeature([], tf.int64),
    "unique_id": tf.io.FixedLenFeature([], tf.int64),
    "input_ids": tf.io.FixedLenFeature([SEQ_LENGTH], tf.int64),
    "input_mask": tf.io.FixedLenFeature([SEQ_LENGTH], tf.int64),
    "segment_ids": tf.io.FixedLenFeature([SEQ_LENGTH], tf.int64),
}
def _decode_record(record, feature_description=None):
    """Decodes a record to a TensorFlow example."""
    feature_description = feature_description or FEATURE_DESCRIPTION
    example = tf.io.parse_single_example(serialized=record, features=feature_description)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for key in [k for k in example.keys() if k not in ['example_id', 'unique_id']]:
        example[key] = tf.cast(example[key], dtype=tf.int32)

    #     example.pop('example_id')
    return example

#%%

if not url_exists(NQ_TEST_TFRECORD_PATH):
    # tf2baseline.F.max_seq_length = 512
    eval_writer = bert_utils.FeatureWriter(filename=NQ_TEST_TFRECORD_PATH,
                                           is_training=False)
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_PATH,
                                           do_lower_case=True)
    features = []
    convert = bert_utils.ConvertExamples2Features(tokenizer=tokenizer,
                                                  is_training=False,
                                                  output_fn=eval_writer.process_feature,
                                                  collect_stat=False)
    n_examples = 0
    # tqdm_notebook = tqdm.tqdm_notebook  # if not on_kaggle_server else None
    for examples in bert_utils.nq_examples_iter(input_file=NQ_TEST_JSONL_PATH,
                                                is_training=False,
                                                tqdm=tqdm):
        for example in examples:
            n_examples += convert(example)
    eval_writer.close()
    print('number of test examples: %d, written to file: %d' % (n_examples, eval_writer.num_features))

#%%

raw_ds = tf.data.TFRecordDataset(NQ_TEST_TFRECORD_PATH)
decoded_ds = raw_ds.map(_decode_record)
batched_ds = decoded_ds.batch(batch_size=BATCH_SIZE, drop_remainder=(TPU is not None))

result = model.predict(batched_ds, verbose=1)

#%%

# add example_id to beginning.
example_id_ds = raw_ds.map(lambda x: tf.io.parse_single_example(
    serialized=x,
    features={"example_id": tf.io.FixedLenFeature([], tf.int64)}
)['example_id'])
result = (np.array(list(example_id_ds)[:len(result[0])]), *result)

#%% md

## 1- Understanding the code
#### For a better understanding, I will briefly explain here.
#### In the item "answer_type", in the last lines of this block, it is responsible for storing the identified response type, which, according to [github project repository](https://github.com/google-research/language/blob/master/language/question_answering/bert_joint/run_nq.py) can be:
UNKNOWN = 0
YES = 1
NO = 2
SHORT = 3
LONG = 4


#%%

def read_candidates_from_one_split(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    print("Reading examples from: %s" % input_path)
    if input_path.endswith(".gz"):
        with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]
    else:
        with tf.io.gfile.GFile(input_path, "r") as input_file:
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]
                # candidates_dict['question'] = e['question_text']
    return candidates_dict


def read_candidates(input_pattern):
    """Read candidates with real multiple processes."""
    input_paths = tf.io.gfile.glob(input_pattern)
    final_dict = {}
    for input_path in input_paths:
        final_dict.update(read_candidates_from_one_split(input_path))
    return final_dict


print("getting candidates...")
candidates_dict = read_candidates('../input/tensorflow2-question-answering/simplified-nq-test.jsonl')

#%%

print("getting result_df...")
result_df = pd.DataFrame({
    "example_id": result[0].squeeze().tolist(),
    "unique_id": result[1].squeeze().tolist(),
    "start_logits": result[2].tolist(),
    "end_logits": result[3].tolist(),
    "answer_type_logits": result[4].tolist()
}).set_index(['example_id', 'unique_id'])
# we pad some instances when using TPU, deduplicate it here.
if TPU is not None:
    print('result_df len before dedup: ' + str(len(result_df)))
    result_df = result_df[~result_df.index.duplicated()]
    print('result_df len after  dedup: ' + str(len(result_df)))

#%%

token_map_ds = raw_ds.map(lambda x: tf.io.parse_single_example(
    serialized=x,
    features={
        "example_id": tf.io.FixedLenFeature([], tf.int64),
        "unique_id": tf.io.FixedLenFeature([], tf.int64),
        # token_map: token to origin map.
        "token_map": tf.io.FixedLenFeature([SEQ_LENGTH], tf.int64)
    }
))
print("getting token_map_df...")
token_map_df = pd.DataFrame.from_records(list(token_map_ds)).applymap(
    lambda x: x.numpy()
).set_index(['example_id', 'unique_id'])
# we pad some instances when using TPU, deduplicate it here.
if TPU is not None:
    print('token_map_df len before: ' + str(len(token_map_df)))
    token_map_df = token_map_df[~token_map_df.index.duplicated()]
    print('token_map_df len before: ' + str(len(token_map_df)))

#%%

joined = result_df.join(token_map_df, on=['example_id', 'unique_id'])

#%%

def best_score_start_end_of_instance(res: pd.Series):
    """
    :param res: index: ['answer_type_logits', 'end_logits', 'start_logits', 'token_map', 'candidates']
    :return: best_score_of_instance, start_short_idx, end_short_idx
    """
    msk_invalid_token = np.array(res['token_map']) == -1
    s_logits, e_logits = pd.Series(res['start_logits']), pd.Series(res['end_logits'])
    # filter logits corresponding to context token and rank top-k.
    s_msk_not_top_k = s_logits.mask(msk_invalid_token) \
                          .rank(method='min', ascending=False) > F['n_best_size']
    s_indexes = np.ma.masked_array(np.arange(s_logits.size),
                                   mask=s_msk_not_top_k | msk_invalid_token)
    e_msk_not_top_k = e_logits.mask(msk_invalid_token) \
                          .rank(method='min', ascending=False) > F['n_best_size']
    e_indexes = np.ma.masked_array(np.arange(e_logits.size),
                                   mask=e_msk_not_top_k | msk_invalid_token)
    # s_e_msk has shape: [512, 512], end index should greater than start index, otherwise, mask it.
    s_e_msk = e_indexes[np.newaxis, :] <= s_indexes[:, np.newaxis]
    # short answer length should litter than max_answer_length, otherwise, mask it.
    s_e_msk |= (e_indexes[np.newaxis, :] - s_indexes[:, np.newaxis] >= F['max_answer_length'])
    # full mask.
    s_e_msk = s_e_msk.filled(True)

    if s_e_msk.all():  # if all start-end combinations has been masked.
        return np.NAN, np.NAN, np.NAN
    else:
        # broadcast to shape: [512, 512], and set mask=s_e_msk
        s_logits_bc = np.ma.array(
            np.broadcast_to(s_logits[:, np.newaxis], shape=[s_logits.size, e_logits.size]),
            mask=s_e_msk)
        e_logits_bc = np.ma.array(
            np.broadcast_to(e_logits[np.newaxis, :], shape=[s_logits.size, e_logits.size]),
            mask=s_e_msk)
        short_span_score = s_logits_bc + e_logits_bc
        cls_token_score = s_logits[0] + e_logits[0]
        score = short_span_score - cls_token_score
        s_short_idx, e_short_idx = divmod(score.argmax(), e_logits.size)

        return score.max(), s_short_idx, e_short_idx


pred_df = pd.DataFrame(columns=['example_id', 'score', 'answer_type',
                                'short_span_start', 'short_span_end',
                                'long_span_start', 'long_span_end', ]
                       ).set_index('example_id')
# fill pred_df
for example_id, group_df in tqdm(joined.groupby('example_id')):
    # group_df: each row got a unique id(unique_id), all rows have a some example_id.
    # columns = ['answer_type_logits', 'end_logits', 'start_logits', 'token_map', 'candidates']
    group_df = group_df.copy().reset_index(level='example_id', drop=True)
    # get best score/start/color and answer type for every instance within same example.
    for u_id, res in group_df.iterrows():
        answer_type_logits = pd.Series(res['answer_type_logits'], index=ANSWER_TYPE_ORDER)
        group_df.loc[u_id, 'ins_answer_type'] = answer_type_logits.idxmax()
        ins_score, ins_start, ins_end = best_score_start_end_of_instance(res)
        group_df.loc[u_id, 'ins_score'] = ins_score
        group_df.loc[u_id, 'ins_short_span_start'] = res['token_map'][ins_start]
        # color span should be exclusive, and np.nan + 1 = np.nan
        group_df.loc[u_id, 'ins_short_span_end'] = res['token_map'][ins_end] + 1
    # we pick instance result who's best score is best among the instances within same example
    best_u_id = group_df['ins_score'].idxmax()
    if best_u_id is not np.NAN:  # if all instances got no score
        short_span_start, short_span_end = group_df.loc[best_u_id, ['ins_short_span_start', 'ins_short_span_end']]
        pred_df.loc[example_id, 'score'] = group_df.loc[best_u_id, 'ins_score']
        pred_df.loc[example_id, 'short_span_start'] = short_span_start
        pred_df.loc[example_id, 'short_span_end'] = short_span_end
        # search for long answer span.
        for cand in candidates_dict[str(example_id)]:
            if cand['top_level'] and cand['start_token'] <= short_span_start and short_span_end <= cand['end_token']:
                pred_df.loc[example_id, 'long_span_start'] = cand['start_token']
                pred_df.loc[example_id, 'long_span_end'] = cand['end_token']
                break
        pred_df.loc[example_id, 'answer_type'] = group_df.loc[best_u_id, 'ins_answer_type']
        # break


#%% md

## 2- Main Change
#### Here is the small, but main change: we created an if to check the predicted response type and thus filter / identify the responses that are passed to the submission file.

#%% md

### Filtering the Answers

#%%

def get_short_pred(pred_row: pd.Series):
    # score(best short answer) is np.NAN means: there's no short/long answers.
    if pred_row['score'] is np.NAN:
        return ''
    # answer_type can not be np.NAN if score is not np.NAN.
    if pred_row['answer_type'] == 'UNKNOWN':
        return ''
    if pred_row['answer_type'] in ['YES', 'NO']:
        return pred_row['answer_type']
    if pred_row['answer_type'] in ['SHORT', 'LONG']:
        if pred_row['score'] < 8:
            return ''
        else:
            return '%d:%d' % (pred_row['short_span_start'], pred_row['short_span_end'])


def get_long_pred(pred_row: pd.Series):
    # score(best short answer) is np.NAN means: there's no short/long answers.
    if pred_row['score'] is np.NAN:
        return ''
    # answer_type can not be np.NAN if score is not np.NAN.
    if pred_row['answer_type'] == 'UNKNOWN':
        return ''
    if pred_row['answer_type'] in ['YES', 'NO', 'SHORT', 'LONG']:
        if pred_row['score'] < 3 or pred_row['long_span_start'] is np.NAN:
            return ''
        else:
            return '%d:%d' % (pred_row['long_span_start'], pred_row['long_span_end'])


#%% md

### Creating a DataFrame

#%%

prediction_df = pred_df.copy()
prediction_df['long_pred'] = pred_df.apply(get_long_pred, axis='columns')
prediction_df['short_pred'] = pred_df.apply(get_short_pred, axis='columns')
prediction_df.index = prediction_df.index.map(lambda x: str(x))

#%% md

### Generating the Submission File

#%%

sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH).set_index('example_id')

for eid, row in prediction_df.iterrows():
    sample_submission.loc[eid + '_long', 'PredictionString'] = row['long_pred']
    sample_submission.loc[eid + '_short', 'PredictionString'] = row['short_pred']

#%%

sample_submission.reset_index().to_csv('submission.csv', index=False)

#%%

prediction_df.tail(60)

#%% md

*Yes
Answers

#%%

yes_answers = sample_submission[sample_submission['PredictionString'] == 'YES']
yes_answers

#%% md

*No
Answers

#%%

no_answers = sample_submission[sample_submission['PredictionString'] == 'NO']
no_answers

#%% md

*Balnk
Answers

#%%

blank_answers = sample_submission[sample_submission['PredictionString'] == '']
blank_answers.head()

#%%

blank_answers.count()

#%% md

### I am only sharing modifications that I believe may help. I left out Tunning and any significant code changes I made.

### We'll be grateful if someone gets a better understanding and can share what really impacts the assessment. No need to share code, just knowledge.
### Thank you!
