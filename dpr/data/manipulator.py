import os
import glob

import tensorflow as tf
from transformers import BertTokenizer

import argparse

from .. import const


def build_tfrecord_text_data_from_jsonl(
    input_path: str,
    out_dir: str,
    records_per_file: int = 5000,
):
    def _record_generator():
        reader = open(input_path, "r")
        while True:
            line = reader.readline()
            if len(line) == 0:
                break

            yield eval(line)

        reader.close()

    def _generate():
        count = 0

        for record in record_generator:
            question = record['question']
            question_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[question.encode(errors='ignore')]))

            positive_ctxs = record['positive_ctxs'][:1]
            if len(positive_ctxs) != 1:
                continue

            hard_negative_ctxs = record['hard_negative_ctxs'][:7]
            if len(hard_negative_ctxs) != 7:
                continue

            positive_ctxs_texts = [ctx['text'].encode(errors='ignore') for ctx in positive_ctxs]
            positive_ctxs_titles = [ctx['title'].encode(errors='ignore') for ctx in positive_ctxs]
            hard_negative_ctxs_texts = [ctx['text'].encode(errors='ignore') for ctx in hard_negative_ctxs]
            hard_negative_ctxs_titles = [ctx['title'].encode(errors='ignore') for ctx in hard_negative_ctxs]

            positive_ctxs_texts_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=positive_ctxs_texts))
            positive_ctxs_titles_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=positive_ctxs_titles))
            hard_negative_ctxs_texts_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=hard_negative_ctxs_texts))
            hard_negative_ctxs_titles_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=hard_negative_ctxs_titles))

            features = {
                'question': question_feature,
                'positive_ctxs/text': positive_ctxs_texts_feature,
                'positive_ctxs/title': positive_ctxs_titles_feature,
                'hard_negative_ctxs/text': hard_negative_ctxs_texts_feature,
                'hard_negative_ctxs/title': hard_negative_ctxs_titles_feature
            }

            example = tf.train.Example(features=tf.train.Features(feature=features))
            example_serialized = example.SerializeToString()
            
            count += 1
            print("Count: {}".format(count))

            yield example_serialized

    record_generator = _record_generator()
    dataset = tf.data.Dataset.from_generator(
        _generate,
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
    tokenizer,
    records_per_file=5000,
    shuffle: bool =  True,
    shuffle_seed: int = 123
):
    text_dataset = load_retriever_tfrecord_text_data(
        input_path=input_path,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed
    )
    int_dataset = transform_retriever_data_from_text_to_int(
        dataset=text_dataset,
        tokenizer=tokenizer
    )

    def _serialize(element):
        question_tensor = element['question_tensor']
        positive_tensor = element['positive_tensor']
        hard_negative_tensor = element['hard_negative_tensor']

        question_tensor_serialized = tf.io.serialize_tensor(tf.io.serialize_sparse(question_tensor))
        positive_tensor_serialized = tf.io.serialize_tensor(tf.io.serialize_sparse(positive_tensor))
        hard_negative_tensor_serialized = tf.io.serialize_tensor(tf.io.serialize_sparse(hard_negative_tensor))

        features = {
            'question_tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[question_tensor_serialized.numpy()])),
            'positive_tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[positive_tensor_serialized.numpy()])),
            'hard_negative_tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hard_negative_tensor_serialized.numpy()]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    def _generate():
        count = 0
        for element in int_dataset:
            count += 1
            print("Count: {}".format(count))
            yield _serialize(element)

    dataset = tf.data.Dataset.from_generator(
        _generate,
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


def load_retriever_tfrecord_text_data(
    input_path: str,
    shuffle: bool = True,
    shuffle_seed: int = 123
):
    dataset = tf.data.Dataset.list_files("{}/*".format(input_path), shuffle=shuffle, seed=shuffle_seed)
    dataset = dataset.interleave(
        lambda dat: tf.data.TFRecordDataset(dat),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    feature_description = {
        'question': tf.io.FixedLenFeature([1,], tf.string),
        'positive_ctxs/text': tf.io.FixedLenFeature([1,], tf.string),
        'positive_ctxs/title': tf.io.FixedLenFeature([1,], tf.string),
        'hard_negative_ctxs/text': tf.io.FixedLenFeature([7,], tf.string),
        'hard_negative_ctxs/title': tf.io.FixedLenFeature([7,], tf.string)
    }

    def _parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(
        _parse_example,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def load_retriever_tfrecord_int_data(
    input_path: str,
    shuffle: bool = True,
    shuffle_seed: int = 123
):
    dataset = tf.data.Dataset.list_files("{}/*".format(input_path), shuffle=shuffle, seed=shuffle_seed)
    dataset = dataset.interleave(
        lambda dat: tf.data.TFRecordDataset(dat),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )
    
    feature_description = {
        'question_tensor': tf.io.FixedLenFeature([1,], tf.string),
        'positive_tensor': tf.io.FixedLenFeature([1,], tf.string),
        'hard_negative_tensor': tf.io.FixedLenFeature([1,], tf.string)
    }
    def _parse(example_proto):
        record = tf.io.parse_single_example(example_proto, feature_description)

        question_serialized = record['question_tensor'][0]
        question_sparse_serialized = tf.io.parse_tensor(question_serialized, out_type=tf.string)
        question_values = tf.io.parse_tensor(question_sparse_serialized[1], out_type=tf.int32)

        positive_serialized = record['positive_tensor'][0]
        positive_sparse_serialized = tf.io.parse_tensor(positive_serialized, out_type=tf.string)
        positive_values = tf.io.parse_tensor(positive_sparse_serialized[1], out_type=tf.int32)

        hard_negative_serialized = record['hard_negative_tensor'][0]
        hard_negative_sparse_serialized = tf.io.parse_tensor(hard_negative_serialized, out_type=tf.string)
        hard_negative_indices = tf.io.parse_tensor(hard_negative_sparse_serialized[0], out_type=tf.int64)
        hard_negative_values = tf.io.parse_tensor(hard_negative_sparse_serialized[1], out_type=tf.int32)
        hard_negative_dense_shape = tf.io.parse_tensor(hard_negative_sparse_serialized[2], out_type=tf.int64)
        hard_negative_tensor = tf.sparse.SparseTensor(
            indices=hard_negative_indices,
            values=hard_negative_values,
            dense_shape=hard_negative_dense_shape
        )

        return {
            'question_tensor': question_values,
            'positive_tensor': positive_values,
            'hard_negative_tensor': hard_negative_tensor
        }

    dataset = dataset.map(
        _parse,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def transform_retriever_data_from_text_to_int(
    dataset: tf.data.Dataset,
    tokenizer
):
    def _transform_ctx_to_tensor(ctx):
        title = ctx[0]
        text = ctx[1]
        title_tokens = tokenizer.tokenize(title)
        text_tokens = tokenizer.tokenize(text)
        sent_tokens = [tokenizer.cls_token] + title_tokens + [tokenizer.sep_token] + text_tokens + [tokenizer.sep_token]
        return tokenizer.convert_tokens_to_ids(sent_tokens)

    def _generate():
        for element in dataset:
            question = element['question'][0].numpy().decode()
            question_tensor = tokenizer.encode(question)
            question_tensor = tf.convert_to_tensor(question_tensor, dtype=tf.int32)
            question_tensor = tf.sparse.from_dense(question_tensor)

            positive_ctxs_text = element['positive_ctxs/text'][0].numpy().decode()
            positive_ctxs_title = element['positive_ctxs/title'][0].numpy().decode()
            positive_ctxs = [positive_ctxs_title, positive_ctxs_text]
            positive_tensor = _transform_ctx_to_tensor(positive_ctxs)
            positive_tensor = tf.convert_to_tensor(positive_tensor, dtype=tf.int32)
            positive_tensor = tf.sparse.from_dense(positive_tensor)

            hard_negative_ctxs_texts = element['hard_negative_ctxs/text'].numpy()
            hard_negative_ctxs_texts = [text.decode() for text in hard_negative_ctxs_texts]
            hard_negative_ctxs_titles = element['hard_negative_ctxs/title'].numpy()
            hard_negative_ctxs_titles = [title.decode() for title in hard_negative_ctxs_titles]
            hard_negative_ctxs = [[title, text] for title, text in zip(hard_negative_ctxs_titles, hard_negative_ctxs_texts)]
            hard_negative_tensor = [_transform_ctx_to_tensor(ctx) for ctx in hard_negative_ctxs]
            hard_negative_tensor = tf.ragged.constant(hard_negative_tensor, dtype=tf.int32).to_sparse()

            yield {
                'question_tensor': question_tensor,
                'positive_tensor': positive_tensor,
                'hard_negative_tensor': hard_negative_tensor
            }

    return tf.data.Dataset.from_generator(
        _generate,
        output_signature={
            'question_tensor': tf.SparseTensorSpec([None], dtype=tf.int32),
            'positive_tensor': tf.SparseTensorSpec([None], dtype=tf.int32),
            'hard_negative_tensor': tf.SparseTensorSpec([7, None], dtype=tf.int32)
        }
    )


def pad(
    dataset: tf.data.Dataset,
    sep_token_id: int = 102,
    max_query_length: int = 256,
    max_context_length: int = 256
):
    def _map(element):
        question_tensor = element['question_tensor']
        question_tensor = question_tensor[:max_query_length] # truncating
        question_tensor = tf.pad(question_tensor, [[0, max_query_length - tf.shape(question_tensor)[0]]]) # padding
        question_tensor = tf.tensor_scatter_nd_update(question_tensor, indices=[[max_query_length - 1]], updates=[sep_token_id])

        positive_tensor = element['positive_tensor']
        positive_tensor = positive_tensor[:max_context_length] # truncating
        positive_tensor = tf.pad(positive_tensor, [[0, max_context_length - tf.shape(positive_tensor)[0]]]) # padding
        positive_tensor = tf.expand_dims(positive_tensor, axis=0)

        hard_negative_tensor = element['hard_negative_tensor']
        hard_negative_tensor = tf.sparse.to_dense(hard_negative_tensor)
        hard_negative_tensor = hard_negative_tensor[:, :max_context_length] # truncating
        hard_negative_tensor = tf.pad(hard_negative_tensor, [[0, 0], [0, max_context_length - tf.shape(hard_negative_tensor)[1]]]) # padding

        contexts = tf.concat([positive_tensor, hard_negative_tensor], axis=0)
        contexts_shape = tf.shape(contexts)
        contexts_scatter = tf.scatter_nd(indices=[[contexts_shape[1] - 1]], updates=[tf.ones(contexts_shape[0], dtype=tf.bool)], shape=[contexts_shape[1], contexts_shape[0]])
        contexts_mask = tf.transpose(contexts_scatter, perm=[1, 0])
        contexts_tensor = tf.where(contexts_mask, sep_token_id, contexts)

        return {
            'question': question_tensor,
            'contexts': contexts_tensor
        }

    return dataset.map(
        _map,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def build_tfrecord_tokenized_data_for_ctx_sources(
    pretrained_model: str,
    ctx_source_path: str,
    out_dir: str,
    max_context_length: int = 256,
    shard_size: int = 42031
):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    ctx_source_files = ["{}/psgs_subset_{:02d}.tsv".format(ctx_source_path, i) for i in range(17)]
    ctx_source_files = tf.data.Dataset.from_tensor_slices(ctx_source_files)
    text_dataset = ctx_source_files.flat_map(
        map_func=lambda x: tf.data.TextLineDataset(x)
    )

    def _transform():
        count = 0
        for element in text_dataset:
            passage = element.numpy().decode()
            id, text, title = passage.split("\t")
            passage_id = "wiki:" + id

            text_tokens = tokenizer.tokenize(text)
            title_tokens = tokenizer.tokenize(title)
            sent_tokens = [tokenizer.cls_token] + title_tokens \
                          + [tokenizer.sep_token] + text_tokens + [tokenizer.sep_token]
            
            if len(sent_tokens) < max_context_length:
                sent_tokens += [tokenizer.pad_token] * (max_context_length - len(sent_tokens))
            
            sent_tokens = sent_tokens[:max_context_length]
            sent_tokens[-1] = tokenizer.sep_token

            context_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
            context_ids = tf.convert_to_tensor(context_ids, dtype=tf.int32)

            count += 1
            print("Count: {}".format(count))

            yield {
                'context_ids': context_ids,
                'passage_id': tf.constant(passage_id)
            }

    dataset = tf.data.Dataset.from_generator(
        _transform,
        output_signature={
            'context_ids': tf.TensorSpec([max_context_length], tf.int32),
            'passage_id': tf.TensorSpec([], tf.string)
        }
    )

    def _serialize(context_ids, passage_id):
        features = {
            'context_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=context_ids)),
            'passage_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_id.numpy()]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    dataset = dataset.map(
        lambda x: tf.py_function(_serialize, inp=[x['context_ids'], x['passage_id']], Tout=tf.string),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    dataset = dataset.window(shard_size)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = 0
    for window in dataset:
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir, "psgs_subset_{:02d}.tfrecord".format(idx)))
        writer.write(window)
        idx += 1


def load_tfrecord_tokenized_data_for_ctx_sources(
    input_path: str,
    max_context_length: int
):
    list_files = glob.glob("{}/*".format(input_path))
    list_files.sort()
    
    dataset = tf.data.Dataset.from_tensor_slices(list_files)
    dataset = dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x)
    )

    feature_description = {
        'context_ids': tf.io.FixedLenFeature([max_context_length], tf.int64),
        'passage_id': tf.io.FixedLenFeature([1], tf.string)
    }
    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(
        _parse,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    def _map_fn(element):
        context_ids = element['context_ids']
        context_ids = tf.cast(context_ids, dtype=tf.int32)
        context_masks = tf.cast(context_ids > 0, tf.bool)
        passage_id = tf.squeeze(element['passage_id'], axis=0)

        return {
            'context_ids': context_ids,
            'context_masks': context_masks,
            'passage_id': passage_id
        }

    dataset = dataset.map(
        _map_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--jsonl-path", type=str, default="data/retriever/nq-train.jsonl")
    parser.add_argument("--tfrecord-text-path", type=str, default="data/retriever/V3/N5000-TEXT")
    parser.add_argument("--tfrecord-int-path", type=str, default="data/retriever/V3/N5000-INT")
    parser.add_argument("--ctx-source-path", type=str, default="gs://openqa-dpr/data/wikipedia_split/shards-42031")
    parser.add_argument("--ctx-tokenized-path", type=str, default="data/wikipedia_split/shards-42031-tfrecord")
    parser.add_argument("--shard-size", type=int, default=42031)
    parser.add_argument("--max-context-length", type=int, default=const.MAX_CONTEXT_LENGTH)
    parser.add_argument("--records-per-file", type=int, default=5000)
    parser.add_argument("--shuffle", type=eval, default=const.SHUFFLE)
    parser.add_argument("--shuffle-seed", type=int, default=const.SHUFFLE_SEED)
    parser.add_argument("--pretrained-model", type=str, default=const.PRETRAINED_MODEL)

    args = parser.parse_args()

    build_tfrecord_tokenized_data_for_ctx_sources(
        ctx_source_path=args.ctx_source_path,
        pretrained_model=args.pretrained_model,
        out_dir=args.ctx_tokenized_path,
        max_context_length=args.max_context_length,
        shard_size=args.shard_size
    )


if __name__ == "__main__":
    main()