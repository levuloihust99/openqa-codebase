import tensorflow as tf
import os
import argparse

from transformers import BertTokenizer

from . import dataloader_v2
from .. import const
from ..utils.tensorizers import Tensorizer

def build_tfrecord_text_data_from_jsonl(
    input_path: str,
    out_dir: str,
    records_per_file: int = 5000,
):
    """Create `.tfrecord` files from `.jsonl` data

    Args:
        input_path (str): Path to the input `.jsonl` file
        out_dir (str): Path to the output `.tfrecord` file 
    """
    def _serialize_ctxs(ctxs):
        ctxs_serialized = []
        for ctx in ctxs:
            ctx_features = {
                'title': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ctx['title'].encode(errors='ignore')])),
                'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ctx['text'].encode(errors='ignore')]))
            }
            ctx_example = tf.train.Example(features=tf.train.Features(feature=ctx_features))
            ctxs_serialized.append(ctx_example.SerializeToString())
        
        return ctxs_serialized
    
    # Convert each record to string
    def _serialize_retriever_sample():
        count = 0
        for record in record_generator:
            question = record['question']
            answers = record['answers']

            positive_ctxs = record['positive_ctxs']
            negative_ctxs = record['negative_ctxs']
            hard_negative_ctxs = record['hard_negative_ctxs']

            features = {
                'question': tf.train.Feature(bytes_list=tf.train.BytesList(value=[question.encode(errors='ignore')])),
                'positive_ctxs': tf.train.Feature(bytes_list=tf.train.BytesList(value=_serialize_ctxs(positive_ctxs))),
                'negative_ctxs': tf.train.Feature(bytes_list=tf.train.BytesList(value=_serialize_ctxs(negative_ctxs))),
                'hard_negative_ctxs': tf.train.Feature(bytes_list=tf.train.BytesList(value=_serialize_ctxs(hard_negative_ctxs)))
            }

            print("Count: {}".format(count + 1))
            count += 1
            yield tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

    def _record_generator():
        reader = open(input_path, "r")
        while True:
            line = reader.readline()
            if len(line) == 0:
                break

            yield eval(line)

        reader.close()

    record_generator = _record_generator()
    dataset = tf.data.Dataset.from_generator(
        _serialize_retriever_sample,
        output_signature=tf.TensorSpec([], tf.string)
    )

    dataset = dataset.window(size=records_per_file)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = 0
    for window in dataset:
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir, "nq-train_{}.tfrecord".format(idx)))
        writer.write(window)
        idx += 1


def build_tfrecord_int_data_from_tfrecord_text_data(
    input_path: str,
    out_dir: str,
    records_per_file=5000,
    shuffle: bool =  True,
    shuffle_seed: int = 123
):
    """Create `.tfrecord` files that contains int data from `.tfrecord` files that contains text data.

    Args:
        input_path (str): Path to the input `.tfrecord` files
        out_dir (str): Path to the directory that contains the output `.tfrecord` files.
    """
    dataset_initial = dataloader_v2.load_retriever_tfrecord_text_data(
        data_path=input_path,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tensorizer = Tensorizer(tokenizer)

    dataset_intermediate = dataloader_v2.transform_retriever_data_from_text_to_int(
        dataset_initial,
        tensorizer
    )

    dataset_final = dataloader_v2.serialize_retriever_int_data(dataset_intermediate)

    dataset = dataset_final.window(records_per_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = 0
    for window in dataset:
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir, "nq-train_{}.tfrecord").format(idx))
        writer.write(window)
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/retriever/V2/N5000-TEXT")
    parser.add_argument("--out-dir", type=str, default="data/retriever/V2/N5000-INT")
    parser.add_argument("--records-per-file", type=int, default=5000)
    parser.add_argument('--shuffle', type=eval, default=const.SHUFFLE)
    parser.add_argument('--shuffle-seed', type=int, default=const.SHUFFLE_SEED)

    args = parser.parse_args()

    input_path = args.input_path
    out_dir = args.out_dir
    records_per_file = args.records_per_file
    shuffle = args.shuffle
    shuffle_seed = args.shuffle_seed

    build_tfrecord_int_data_from_tfrecord_text_data(
        input_path=input_path,
        out_dir=out_dir,
        records_per_file=records_per_file,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed
    )