"""
This module is tend for running as module, i.e python -m dpr.databuilder
"""

import numpy as np
import time
import os
import json
import argparse
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from . import dataloader


def main(called_as_module: bool = False, **kwargs):
    global args
    global raw_dataset

    if called_as_module:
        assert kwargs
        args = Namespace(**kwargs)

    path = args.input_path
    abs_path = os.path.abspath(path)
    root_dir = abs_path[:abs_path.index(args.input_path) - 1]

    output_dir = os.path.join(root_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.build_type == 0:
        raw_dataset = tf.data.TextLineDataset(path)
        dataset = tf.data.Dataset.from_generator(_generate, \
                                                output_signature={
                                                    'question': tf.TensorSpec((), tf.string),
                                                    'positive_ctx': tf.TensorSpec((2), tf.string),
                                                    'hard_negative_ctx': tf.TensorSpec((2), tf.string),
                                                })

        dataset = dataset.map(
            lambda x: tf.py_function(func=_serialize_text_tensor, inp=[x['question'], x['positive_ctx'], x['hard_negative_ctx']], Tout=tf.string),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.window(args.records_per_file)

        idx = 0
        for window in dataset:
            writer = tf.data.experimental.TFRecordWriter(os.path.join(output_dir, 'nq-train_{}.tfrecord'.format(idx)))
            writer.write(window)
            idx += 1
            if idx == args.max_files:
                break

    elif args.build_type == 1:
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_model)
        dataset = dataloader.load_retriever_text_data(path)
        dataset = dataloader.transform_retriever_text_data_to_tensors(dataset, tokenizer)

        dataset = dataset.map(
            lambda x: tf.py_function(func=_serialize_int_tensor,
                                    inp=[x['question_tensor'][0], x['positive_tensor'][0], x['hard_negative_tensor'][0]], 
                                    Tout=tf.string),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.window(args.records_per_file)

        idx = 0
        for window in dataset:
            writer = tf.data.experimental.TFRecordWriter(os.path.join(output_dir, 'nq-train_{}.tfrecord'.format(idx)))
            writer.write(window)
            idx += 1

    else:
        raw_dataset = tf.data.TextLineDataset(path)
        dataset = tf.data.Dataset.from_generator(_generate, \
                                                output_signature={
                                                    'question': tf.TensorSpec((), tf.string),
                                                    'positive_ctx': tf.TensorSpec((2), tf.string),
                                                    'hard_negative_ctx': tf.TensorSpec((2), tf.string),
                                                })

        tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_model)
        dataset = dataloader.transform_retriever_text_data_to_tensors(dataset, tokenizer)
        dataset = dataset.map(
            lambda x: tf.py_function(func=_serialize_int_tensor,
                                    inp=[x['question_tensor'][0], x['positive_tensor'][0], x['hard_negative_tensor'][0]], 
                                    Tout=tf.string),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.window(args.records_per_file)

        idx = 0
        for window in dataset:
            writer = tf.data.experimental.TFRecordWriter(os.path.join(output_dir, 'nq-train_{}.tfrecord'.format(idx)))
            writer.write(window)
            idx += 1


def _generate():
    assert 'raw_dataset' in globals(), "You must define `raw_dataset` before running this function"

    count = 0
    actual = 0
    for element in raw_dataset:

        element = eval(element.numpy())
        question = element['question']
        positive_ctxs = element['positive_ctxs'][:1]
        hard_negative_ctxs = element['hard_negative_ctxs'][:1]

        if len(positive_ctxs) == 0 or len(hard_negative_ctxs) == 0:
            continue

        positive_ctx = positive_ctxs[0]
        hard_negative_ctx = hard_negative_ctxs[0]

        positive_ctx = [positive_ctx['title'], positive_ctx['text']]
        hard_negative_ctx = [hard_negative_ctx['title'], hard_negative_ctx['text']]

        count += 1
        count_out = "Count: {:05d}".format(count)

        try:
            [st.encode('utf-8') for st in positive_ctx]
            [st.encode('utf-8') for st in hard_negative_ctx]
        except UnicodeEncodeError:
            continue

        actual += 1
        actual_out = "Actual: {:05d}".format(actual)

        print("{} --- {}".format(count_out, actual_out))

        yield {'question': question, 'positive_ctx': positive_ctx, 'hard_negative_ctx': hard_negative_ctx}


def _bytes_feature(value):
    if isinstance(value, list):
        value = [v.encode('utf-8') for v in value]
    elif isinstance(value, np.ndarray):
        pass
    elif isinstance(value, bytes):
        value = [value]
    else:
        value = [value.encode('utf-8')]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _serialize_text_tensor(question, positive_ctx, hard_negative_ctx):
    features = {
        'question': _bytes_feature(question.numpy()),
        'positive_ctx': _bytes_feature(positive_ctx.numpy()),
        'hard_negative_ctx': _bytes_feature(hard_negative_ctx.numpy()),
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def _serialize_int_tensor(question_tensor, positive_tensor, hard_negative_tensor):
    features = {
        'question_tensor': _int64_feature(question_tensor),
        'positive_tensor': _int64_feature(positive_tensor),
        'hard_negative_tensor': _int64_feature(hard_negative_tensor)
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='data/retriever/N5000/*')
    parser.add_argument('--output-dir', type=str, default='data/retriever/N5000-INT', help='Directory for storing output `.tfrecord` file')
    parser.add_argument('--build-type', type=int, default=1, help="0 means building from `.jsonl` to `.tfrecord`;" \
                        "1 means building from `.tfrecord` (text data) to `.tfrecord` (int data)")
    parser.add_argument('--max-files', type=int, default=-1)
    parser.add_argument('--records-per-file', type=int, default=5000)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--bert-pretrained-model", type=str, default='bert-base-uncased')
    args = parser.parse_args()

    main()
