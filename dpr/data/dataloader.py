import tensorflow as tf
from transformers import BertTokenizer
from typing import List, Dict, Tuple, Text, Optional


def load_retriever_text_data(
    data_path: str,
    shuffle: bool = True,
    shuffle_seed: int = 123,
):
    dataset = tf.data.Dataset.list_files(data_path, shuffle=shuffle, seed=shuffle_seed)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    feature_description = {
        'question': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'positive_ctx': tf.io.FixedLenFeature([2], tf.string, default_value=['', '']),
        'hard_negative_ctx': tf.io.FixedLenFeature([2], tf.string, default_value=['', ''])
    }


    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(
        _parse,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def transform_retriever_text_data_to_tensors(
    dataset: tf.data.Dataset,
    tokenizer: BertTokenizer,
):
    """Transform retriever data in text format to tensor format

    Args:
        dataset (tf.data.Dataset): Input dataset, whose elements are in text format
        tokenizer (BertTokenizer): Tokenizer used to tokenize text to tokens
        max_sequence_length (int): Max length of the context sequence. Truncate the sequence if length exceeds `max_sequence_length`
    """

    def _text_to_ids(passage: tf.Tensor) -> List[Text]:
        title = passage[0].numpy().decode('utf-8')
        text = passage[1].numpy().decode('utf-8')

        tokenized_title = tokenizer.tokenize(title)
        tokenized_text = tokenizer.tokenize(text)

        tokens = ['[CLS]'] + tokenized_title + ['[SEP]'] + tokenized_text + ['[SEP]']

        return tokenizer.convert_tokens_to_ids(tokens)

    def _generate():
        count = 0
        for element in dataset:

            question = element['question'].numpy()
            question = tokenizer.encode(question.decode('utf-8'))
            question_tensor = tf.constant(question)

            positive_ctx = _text_to_ids(element['positive_ctx'])
            hard_negative_ctx = _text_to_ids(element['hard_negative_ctx'])

            positive_tensor = tf.constant(positive_ctx)
            hard_negative_tensor = tf.constant(hard_negative_ctx)

            count += 1
            print(f"Count: {count:09d}")

            question_tensor = tf.expand_dims(question_tensor, axis=0)
            positive_tensor = tf.expand_dims(positive_tensor, axis=0)
            hard_negative_tensor = tf.expand_dims(hard_negative_tensor, axis=0)

            yield {
                'question_tensor': tf.RaggedTensor.from_tensor(question_tensor, padding=0),
                'positive_tensor': tf.RaggedTensor.from_tensor(positive_tensor, padding=0),
                'hard_negative_tensor': tf.RaggedTensor.from_tensor(hard_negative_tensor, padding=0),
            }

    return tf.data.Dataset.from_generator(
        _generate,
        output_signature={
            'question_tensor': tf.RaggedTensorSpec(shape=[1, None], dtype=tf.int32),
            'positive_tensor': tf.RaggedTensorSpec(shape=[1, None], dtype=tf.int32),
            'hard_negative_tensor': tf.RaggedTensorSpec(shape=[1, None], dtype=tf.int32),
        }
    )


def load_retriever_int_data(
    data_path: str,
    shuffle: bool = True,
    shuffle_seed: int = 123
):
    dataset = tf.data.Dataset.list_files(data_path, shuffle=shuffle, seed=shuffle_seed)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    feature_description = {
        'question_tensor': tf.io.RaggedFeature(tf.int64),
        'positive_tensor': tf.io.RaggedFeature(tf.int64),
        'hard_negative_tensor': tf.io.RaggedFeature(tf.int64)
    }

    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(
        _parse,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        lambda x: {k: tf.cast(v, dtype=tf.int32) for k, v in x.items()},
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def merge_contexts(dataset: tf.data.Dataset):
    """Merge positive and negative context. Dataset must be batched"""

    def _map_fn(element):

        question_ids = element['question_tensor']
        positive_tensor = element['positive_tensor']
        hard_negative_tensor = element['hard_negative_tensor']

        merge_tensor = tf.stack([positive_tensor, hard_negative_tensor], axis=0)
        merge_tensor = tf.transpose(merge_tensor, perm=[1, 0, 2])
        context_ids = tf.reshape(merge_tensor, [-1, 256])

        question_masks = tf.cast(question_ids > 0, tf.int32)
        context_masks = tf.cast(context_ids > 0, tf.int32)

        return {
            'question_ids': question_ids,
            'question_masks': question_masks,
            'context_ids': context_ids,
            'context_masks': context_masks
        }

    return dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def pad(
    dataset: tf.data.Dataset, 
    max_query_length: int = 32, 
    max_context_length: int = 256
):
    """Pad the input sequence to max sequence length. Input must be non-batched"""
    def _map_fn(element):
        question_tensor = element['question_tensor']
        positive_tensor = element['positive_tensor']
        hard_negative_tensor = element['hard_negative_tensor']

        question_tensor = tf.pad(question_tensor, [[0, max_query_length]])[:max_query_length]
        positive_tensor = tf.pad(positive_tensor, [[0, max_context_length]])[:max_context_length]
        hard_negative_tensor = tf.pad(hard_negative_tensor, [[0, max_context_length]])[:max_context_length]

        return {
            'question_tensor': question_tensor,
            'positive_tensor': positive_tensor,
            'hard_negative_tensor': hard_negative_tensor
        }

    return dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
