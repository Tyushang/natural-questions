# %% [markdown]
# # Using Tensorflow 2.0 with Bert on Natural Questions
# 
# In this notebook, we'll be using the Bert baseline for Tensorflow to create predictions for the Natural Questions test set. Note that this uses a model that has already been pre-trained - we're only doing inference here. A GPU is required, and this should take between 1-2 hours to run.
# 
# The original script can be found [here](https://github.com/google-research/language/blob/master/language/question_answering/bert_joint/run_nq.py).
# The supporting modules were drawn from the [official Tensorflow model repository](https://github.com/tensorflow/models/tree/master/official/nlp). The bert-joint-baseline data is described [here](https://github.com/google-research/language/tree/master/language/question_answering/bert_joint).
# 
# **Note:** This baseline uses code that was migrated from TF1.x. Be aware that it contains use of tf.compat.v1, which is not permitted to be eligible for [TF2.0 prizes in this competition](https://www.kaggle.com/c/tensorflow2-question-answering/overview/prizes). It is intended to be used as a starting point, but we're excited to see how much better you can do using TF2.0!

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tf2_0_baseline_w_bert as tf2baseline
import bert_modeling as modeling
import bert_optimization as optimization
import bert_tokenization as tokenization
import json

tf.compat.v1.disable_eager_execution()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# In this case, we've got some extra BERT model files under `../input/bertjointbaseline`

import os

for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pprint
with tf.compat.v1.Session('tyushang') as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())


# Any results you write to the current directory are saved as output.

# %% [markdown]
# #### Tensorflow flags are variables that can be passed around within the TF system. Every flag below has some context provided regarding what the flag is and how it's used.
# 
# #### Most of these can be changed as desired, with the exception of the Special Flags at the bottom, which _must_ stay as-is to work with the Kaggle back color.

# %% [code]
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

# flags = tf.compat.v1.app.flags
from tensorflow_core.python.platform import flags

del_all_flags(tf.compat.v1.app.flags.FLAGS)


# if not define 'mode' & 'port' when run in python/ipython interactive shell, error will be raised:
# absl.flags._exceptions.UnrecognizedFlagError: Unknown command line flag 'mode' / 'port'
flags.DEFINE_string('mode', 'default', 'help')
flags.DEFINE_string('port', 'default', 'help')

flags.DEFINE_string(
    "bert_config_file", "../input/bertjointbaseline/bert_config.json",
    "The bert_config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "../input/bertjointbaseline/vocab-nq.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "outdir",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

flags.DEFINE_string(
    "output_prediction_file", "predictions.json",
    "Where to print predictions in NQ prediction format, to be passed to"
    "natural_questions.nq_eval.")

flags.DEFINE_string(
    "init_checkpoint", "../input/bertjointbaseline/bert_joint.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "verbosity", 1, "How verbose our error messages should be")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the s_short_idx and color predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.TPU:8470 "
    "url.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal NQ evaluation.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

## Special flags - do not change

flags.DEFINE_string(
    "predict_file", "../input/tensorflow2-question-answering/simplified-nq-test.jsonl",
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_boolean("logtostderr", True, "Logs to stderr")
flags.DEFINE_boolean("undefok", True, "it's okay to be undefined")
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('HistoryManager.hist_file', '', 'kernel')

FLAGS = flags.FLAGS

# %% [markdown]
# #### Here, we:
# 1. Set up Bert
# 2. Read in the test set
# 3. Run it past the pre-built Bert model to create embeddings
# 4. Use those embeddings to make predictions
# 5. Write those predictions to `predictions.json`
# 
# Feel free to change the code below. Code for the `tf2baseline.*` functions is included in the `tf2_0_baseline_w_bert` utility script, and can be customized, whether by forking the utility script and updating it, or by creating your own non-`tf2baseline` versions in this kernel.
# 
# Note: the `tf2_0_baseline_w_bert` utility script contains code for training your own embeddings. Here that code is removed.

# %% [code]
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

tf2baseline.validate_flags_or_throw(bert_config)
tf.io.gfile.makedirs(FLAGS.output_dir)

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

tpu_cluster_resolver = None
if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.compat.v1.estimator.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

num_train_steps = None
num_warmup_steps = None

model_fn = tf2baseline.model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu)

# If TPU is not available, this falls back to normal Estimator on CPU or GPU.
estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)

if FLAGS.do_predict:
    if not FLAGS.output_prediction_file:
        raise ValueError(
            "--output_prediction_file must be defined in predict mode.")

    eval_examples = tf2baseline.read_nq_examples(
        input_file=FLAGS.predict_file, is_training=False, n_lines=10)

    print("F.predict_file", FLAGS.predict_file)

    eval_writer = tf2baseline.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []


    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)


    num_spans_to_ids = tf2baseline.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()
    eval_filename = eval_writer.filename

    print("***** Running predictions *****")
    print(f"  Num orig examples = %d" % len(eval_examples))
    print(f"  Num split examples = %d" % len(eval_features))
    print(f"  Batch size = %d" % FLAGS.predict_batch_size)
    for spans, ids in num_spans_to_ids.items():
        print(f"  Num split into %d = %d" % (spans, len(ids)))

    predict_input_fn = tf2baseline.input_fn_builder(
        input_file=eval_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    print(eval_filename)

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []

    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            print("Processing example: %d" % (len(all_results)))

        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]

        all_results.append(
            tf2baseline.RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))

    print("Going to candidates file")

    candidates_dict = tf2baseline.read_candidates(FLAGS.predict_file)

    print("setting up eval features")

    eval_features = [
        tf.train.Example.FromString(r)
        for r in tf.compat.v1.python_io.tf_record_iterator(eval_filename)
    ]

    print("compute_pred_dict")

    nq_pred_dict = tf2baseline.compute_pred_dict(candidates_dict, eval_features,
                                                 [r._asdict() for r in all_results])
    predictions_json = {"predictions": list(nq_pred_dict.values())}

    print("writing json")

    with tf.io.gfile.GFile(FLAGS.output_prediction_file, "w") as f:
        json.dump(predictions_json, f, indent=4)

# %% [markdown]
# #### Now, we turn `predictions.json` into a `submission.csv` file.

# %% [code]
test_answers_df = pd.read_json("/kaggle/working/predictions.json")


# %% [markdown]
# The Bert model produces a `confidence` score, which the Kaggle metric does not use. You, however, can use that score to determine which answers get submitted. See the limits commented out in `create_short_answer` and `create_long_answer` below for an example.
#
# Values for `confidence` will range between `1.0` and `2.0`.

# %% [code]
def create_short_answer(entry):
    # if entry["short_answers_score"] < 1.5:
    #     return ""

    answer = []
    for short_answer in entry["short_answers"]:
        if short_answer["start_token"] > -1:
            answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
    if entry["yes_no_answer"] != "NONE":
        answer.append(entry["yes_no_answer"])
    return " ".join(answer)


def create_long_answer(entry):
    # if entry["long_answer_score"] < 1.5:
    # return ""

    answer = []
    if entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
    return " ".join(answer)


# %% [code]
test_answers_df["long_answer_score"] = test_answers_df["predictions"].apply(lambda q: q["long_answer_score"])
test_answers_df["short_answer_score"] = test_answers_df["predictions"].apply(lambda q: q["short_answers_score"])

# %% [code]
test_answers_df["long_answer_score"].describe()

# %% [markdown]
# An example of what each sample's answers look like in `prediction.json`:

# %% [code]
test_answers_df.predictions.values[0]

# %% [markdown]
# We re-format the JSON answers to match the requirements for submission.

# %% [code]
test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)
test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)
test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))

long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

# %% [markdown]
# Then we add them to our sample submission. Recall that each sample has both a `_long` and `_short` entry in the sample submission, one for each type of answer.

# %% [code]
sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")

long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(
    lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(
    lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)

sample_submission.loc[
    sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings
sample_submission.loc[
    sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings

# %% [markdown]
# And finally, we write out our submission!

# %% [code]
sample_submission.to_csv("submission.csv", index=False)

