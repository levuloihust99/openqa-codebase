import tensorflow as tf
import pickle
import os
import sys
import glob
import logging
import argparse


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def build_tfrecord_reader_train_data(
    dpr_reader_data_path: str,
    out_dir: str,
    records_per_file: int = 5000,
):
    sys.path = ["/media/levuloi/Storage/thesis/DPR"] + sys.path
    from dpr.data.reader_data import ReaderSample
    sys.path.pop(0)
    
    files = glob.glob(os.path.join(dpr_reader_data_path, "train_*.pkl"))
    
    def _generator():
        # count = 0
        for f in files:
            with open(f, "rb") as reader:
                data = pickle.load(reader)
            for sample in data:
                yield sample
                # count += 1
                # print("Load #{} samples".format(count))

    generator = _generator()

    def _generate():
        # count = 0
        for sample in generator:
            answers = sample.answers
            question = sample.question

            positive_passages = sample.positive_passages
            negative_passages = sample.negative_passages
            if len(positive_passages) == 0 or len(negative_passages) == 0:
                continue

            positive_sequence_ids = [psg.sequence_ids.tolist() for psg in positive_passages]
            positive_offsets = [psg.passage_offset for psg in positive_passages]
            negative_sequence_ids = [psg.sequence_ids.tolist() for psg in negative_passages]
            negative_offsets = [psg.passage_offset for psg in negative_passages]

            list_answers_spans = [psg.answers_spans for psg in positive_passages]
            list_start_positions = []
            list_end_positions = []
            for answers_spans in list_answers_spans:
                start_positions, end_positions = zip(*answers_spans)
                list_start_positions.append(list(start_positions))
                list_end_positions.append(list(end_positions))
            
            answers = tf.sparse.from_dense(answers)
            question = tf.constant(question)
            positive_sequence_ids = tf.ragged.constant(positive_sequence_ids, dtype=tf.int32).to_sparse()
            positive_offsets = tf.sparse.from_dense(tf.convert_to_tensor(positive_offsets, dtype=tf.int32))
            list_start_positions  = tf.ragged.constant(list_start_positions, dtype=tf.int32).to_sparse()
            list_end_positions    = tf.ragged.constant(list_end_positions, dtype=tf.int32).to_sparse()
            negative_sequence_ids = tf.ragged.constant(negative_sequence_ids, dtype=tf.int32).to_sparse()
            negative_offsets = tf.sparse.from_dense(tf.convert_to_tensor(negative_offsets, dtype=tf.int32))

            yield {
                "answers": answers,
                "question": question,
                "positive_passages/sequence_ids": positive_sequence_ids,
                "positive_passages/passage_offset": positive_offsets,
                "positive_passages/start_positions": list_start_positions,
                "positive_passages/end_positions": list_end_positions,
                "negative_passages/sequence_ids": negative_sequence_ids,
                "negative_passages/passage_offset": negative_offsets
            }

            # count += 1
            # print("Generate {} tensors".format(count))

    tensor_dataset = tf.data.Dataset.from_generator(
        _generate,
        output_signature={
            "answers": tf.SparseTensorSpec([None], dtype=tf.string),
            "question": tf.TensorSpec([], dtype=tf.string),
            "positive_passages/sequence_ids": tf.SparseTensorSpec([None, None], dtype=tf.int32),
            "positive_passages/passage_offset": tf.SparseTensorSpec([None], dtype=tf.int32),
            "positive_passages/start_positions": tf.SparseTensorSpec([None, None], dtype=tf.int32),
            "positive_passages/end_positions": tf.SparseTensorSpec([None, None], dtype=tf.int32),
            "negative_passages/sequence_ids": tf.SparseTensorSpec([None, None], dtype=tf.int32),
            "negative_passages/passage_offset": tf.SparseTensorSpec([None], dtype=tf.int32)
        }
    )

    def _serialize(element):
        answers = element['answers']
        serialized_answers = tf.io.serialize_tensor(tf.io.serialize_sparse(answers))
        answers_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_answers.numpy()]))

        question = element['question']
        question_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[question.numpy()]))

        positive_sequence_ids = element["positive_passages/sequence_ids"]
        serialized_positive_sequence_ids = tf.io.serialize_tensor(tf.io.serialize_sparse(positive_sequence_ids))
        positive_sequence_ids_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_positive_sequence_ids.numpy()]))

        positive_offsets = element['positive_passages/passage_offset']
        serialized_positive_offsets = tf.io.serialize_tensor(tf.io.serialize_sparse(positive_offsets))
        positive_offsets_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_positive_offsets.numpy()]))

        positive_start_positions = element['positive_passages/start_positions']
        serialized_positive_start_positions = tf.io.serialize_tensor(tf.io.serialize_sparse(positive_start_positions))
        positive_start_positions_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_positive_start_positions.numpy()]))

        positive_end_positions = element['positive_passages/end_positions']
        serialized_positive_end_positions = tf.io.serialize_tensor(tf.io.serialize_sparse(positive_end_positions))
        positive_end_positions_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_positive_end_positions.numpy()]))

        negative_sequence_ids = element['negative_passages/sequence_ids']
        serialized_negative_sequence_ids = tf.io.serialize_tensor(tf.io.serialize_sparse(negative_sequence_ids))
        negative_sequence_ids_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_negative_sequence_ids.numpy()]))

        negative_offsets = element['negative_passages/passage_offset']
        serialized_negative_offsets = tf.io.serialize_tensor(tf.io.serialize_sparse(negative_offsets))
        negative_offsets_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_negative_offsets.numpy()]))

        features = {
            "answers": answers_feature,
            "question": question_feature,
            "positive_passages/sequence_ids": positive_sequence_ids_feature,
            "positive_passages/passage_offset": positive_offsets_feature,
            "positive_passages/start_positions": positive_start_positions_feature,
            "positive_passages/end_positions": positive_end_positions_feature,
            "negative_passages/sequence_ids": negative_sequence_ids_feature,
            "negative_passages/passage_offset": negative_offsets_feature
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    def _generate_serialized():
        count = 0
        for element in tensor_dataset:
            yield _serialize(element)
            count += 1
            print("Serialize {} tensors".format(count))
            print("---------------------------------------------------------------------------------------------------------------------------------------------")
        
    dataset = tf.data.Dataset.from_generator(
        _generate_serialized,
        output_signature=tf.TensorSpec([], tf.string)
    )

    dataset = dataset.window(records_per_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = 0
    for window in dataset:
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir, "reader_data_{:02d}.tfrecord".format(idx)))
        writer.write(window)
        idx += 1


def load_tfrecord_reader_train_data(
    input_path: str,
):
    list_files = tf.io.gfile.listdir(input_path)
    list_files.sort()
    list_files = [os.path.join(input_path, f) for f in list_files]

    file_dataset = tf.data.Dataset.from_tensor_slices(list_files)
    serialized_dataset = file_dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x)
    )

    feature_description = {
        "answers": tf.io.FixedLenFeature([1], tf.string),
        "question": tf.io.FixedLenFeature([1], tf.string),
        "positive_passages/sequence_ids": tf.io.FixedLenFeature([1], tf.string),
        "positive_passages/passage_offset": tf.io.FixedLenFeature([1], tf.string),
        "positive_passages/start_positions": tf.io.FixedLenFeature([1], tf.string),
        "positive_passages/end_positions": tf.io.FixedLenFeature([1], tf.string),
        "negative_passages/sequence_ids": tf.io.FixedLenFeature([1], tf.string),
        "negative_passages/passage_offset": tf.io.FixedLenFeature([1], tf.string)
    }
    def _parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)
    
    tensor_dataset = serialized_dataset.map(
        _parse_example,
        num_parallel_calls=True,
        deterministic=True
    )

    return tensor_dataset


def transform_to_reader_train_dataset(
    dataset: tf.data.Dataset,
    max_sequence_length: int = 256,
    max_answers: int = 10
):
    def _pad(element):
        # parse positive sequence ids
        positive_sequence_ids_serialized = element['positive_passages/sequence_ids'][0]
        positive_sequence_ids_sparse = tf.io.parse_tensor(positive_sequence_ids_serialized, out_type=tf.string)
        positive_sequence_ids_indices = tf.io.parse_tensor(positive_sequence_ids_sparse[0], out_type=tf.int64)
        positive_sequence_ids_values = tf.io.parse_tensor(positive_sequence_ids_sparse[1], out_type=tf.int32)
        positive_sequence_ids_dense_shape = tf.io.parse_tensor(positive_sequence_ids_sparse[2], out_type=tf.int64)
        positive_sequence_ids = tf.sparse.SparseTensor(
            indices=positive_sequence_ids_indices,
            values=positive_sequence_ids_values,
            dense_shape=positive_sequence_ids_dense_shape
        )
        positive_sequence_ids = tf.sparse.to_dense(positive_sequence_ids)

        # pad positive sequence ids
        positive_sequence_ids = positive_sequence_ids[:, :max_sequence_length] # truncate
        positive_sequence_ids = tf.pad(positive_sequence_ids, [[0, 0], [0, max_sequence_length - tf.shape(positive_sequence_ids)[1]]]) # padding

        # parse start positions
        positive_start_positions_serialized = element['positive_passages/start_positions'][0]
        positive_start_positions_sparse = tf.io.parse_tensor(positive_start_positions_serialized, out_type=tf.string)
        positive_start_positions_indices = tf.io.parse_tensor(positive_start_positions_sparse[0], out_type=tf.int64)
        positive_start_positions_values = tf.io.parse_tensor(positive_start_positions_sparse[1], out_type=tf.int32)
        positive_start_positions_dense_shape = tf.io.parse_tensor(positive_start_positions_sparse[2], out_type=tf.int64)
        positive_start_positions = tf.sparse.SparseTensor(
            indices=positive_start_positions_indices,
            values=positive_start_positions_values,
            dense_shape=positive_start_positions_dense_shape
        )
        positive_start_positions = tf.sparse.to_dense(positive_start_positions)

        # pad start positions
        positive_start_positions = positive_start_positions[:, :max_answers] # truncate
        positive_start_positions = tf.pad(positive_start_positions, [[0, 0], [0, max_answers - tf.shape(positive_start_positions)[1]]]) # padding

        # parse positive end positions
        positive_end_positions_serialized = element['positive_passages/end_positions'][0]
        positive_end_positions_sparse = tf.io.parse_tensor(positive_end_positions_serialized, out_type=tf.string)
        positive_end_positions_indices = tf.io.parse_tensor(positive_end_positions_sparse[0], out_type=tf.int64)
        positive_end_positions_values = tf.io.parse_tensor(positive_end_positions_sparse[1], out_type=tf.int32)
        positive_end_positions_dense_shape = tf.io.parse_tensor(positive_end_positions_sparse[2], out_type=tf.int64)
        positive_end_positions = tf.sparse.SparseTensor(
            indices=positive_end_positions_indices,
            values=positive_end_positions_values,
            dense_shape=positive_end_positions_dense_shape
        )
        positive_end_positions = tf.sparse.to_dense(positive_end_positions)

        # pad end positions
        positive_end_positions = positive_end_positions[:, :max_answers] # truncate
        positive_end_positions = tf.pad(positive_end_positions, [[0, 0], [0, max_answers - tf.shape(positive_end_positions)[1]]])

        return {
            "input_ids": positive_sequence_ids,
            "start_positions": positive_start_positions,
            "end_positions": positive_end_positions
        }

    dataset = dataset.map(
        _pad,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    dataset = dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x),
    )

    return dataset


def build_tfrecord_reader_dev_dataset(
    dpr_reader_data_path: str,
    out_dir: str
):
    sys.path = ["/media/levuloi/Storage/thesis/DPR"] + sys.path
    from dpr.data.reader_data import ReaderSample
    sys.path.pop(0)

    with open(dpr_reader_data_path, "rb") as reader:
        dev_data = pickle.load(reader)

    def _generate():
        for sample in dev_data:
            answers  = sample.answers
            answers  = tf.sparse.from_dense(answers, dtype=tf.string)

            question = sample.question
            question = tf.constant(question, dtype=tf.string)

            passages = sample.passages
            passages_sequence_ids = [psg.sequence_ids for psg in passages]
            passages_sequence_ids = tf.ragged.constant(passages_sequence_ids, dtype=tf.int32).to_sparse()
            passages_offsets = [psg.passage_offset for psg in passages]
            passages_offsets = tf.sparse.from_dense(tf.convert_to_tensor(passages_offsets, dtype=tf.int32))
            
            yield {
                "answers": answers,
                "question": question,
                "passages/sequence_ids": passages_sequence_ids,
                "passages/passage_offset": passages_offsets
            }

    tensor_dataset = tf.data.Dataset.from_generator(
        _generate,
        output_signature={
            "answers": tf.SparseTensorSpec(shape=[None], dtype=tf.string),
            "question": tf.TensorSpec(shape=[], dtype=tf.string),
            "passages/sequence_ids": tf.SparseTensorSpec(shape=[None, None], dtype=tf.int32),
            "passages/passage_offset": tf.SparseTensorSpec(shape=[None], dtype=tf.int32)
        }
    )

    def _serialize(element):
        answers = element['answers']
        answers_serialized = tf.io.serialize_tensor(tf.io.serialize_sparse(answers))
        answers_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[answers_serialized.numpy()]))

        question = element['question']
        question_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[question.numpy()]))

        passages_sequence_ids = element['passages/sequence_ids']
        passages_sequence_ids_serialized = tf.io.serialize_tensor(tf.io.serialize_sparse(passages_sequence_ids))
        passages_sequence_ids_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[passages_sequence_ids_serialized.numpy()]))

        passages_offsets = element['passages/passage_offset']
        passages_offsets_serialized = tf.io.serialize_tensor(tf.io.serialize_sparse(passages_offsets))
        passages_offsets_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[passages_offsets_serialized.numpy()]))

        features = {
            "answers": answers_feature,
            "question": question_feature,
            "passages/sequence_ids": passages_sequence_ids,
            "passages/passage_offset": passages_offsets_feature
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    def _generate_serialized():
        count = 0
        for element in tensor_dataset:
            yield _serialize(element)
            count += 1
            print("Serialize {} samples".format(count))
            print("---------------------------------------------------------------------------------------------------------------------------------------------")

    dataset = tf.data.Dataset.from_generator(
        _generate_serialized,
        output_signature=tf.TensorSpec([], dtype=tf.string)
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir, "dev.tfrecord"))
    writer.write(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpr-reader-data-path", type=str, default="/media/levuloi/Storage/thesis/DPR/downloads/data/reader/nq/single")
    parser.add_argument("--out-dir", type=str, default="data/reader/nq/train")
    parser.add_argument("--records-per-file", type=int, default=5000)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--max-answers", type=int, default=10)
    parser.add_argument("--input-path", type=str, default="data/reader/nq/train")

    args = parser.parse_args()

    # build_tfrecord_reader_data(
    #     dpr_reader_data_path=args.dpr_reader_data_path,
    #     out_dir=args.out_dir,
    #     records_per_file=args.records_per_file,
    # )

    # print("Generating tfrecord data for reader done")

    dataset = load_tfrecord_reader_train_data(
        input_path=args.input_path,
    )

    dataset = transform_to_reader_train_dataset(
        dataset,
        max_sequence_length=args.max_sequence_length,
        max_answers=args.max_answers
    )

    count = 0
    iterator = iter(dataset)
    for element in dataset:
        count += 1
        print("Count {}".format(count))

    print("********************************************************")
    print("Train data size: {}".format(count))

    print("done")