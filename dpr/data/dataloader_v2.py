import tensorflow as tf
import time
import argparse
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

from transformers.models.bert.tokenization_bert import BertTokenizer

from ..utils.tensorizers import Tensorizer


def load_retriever_tfrecord_text_data(
    data_path: str,
    shuffle: bool = True,
    shuffle_seed: int = 123
):
    """Deserialize text data from `.tfrecord` file

    Args:
        input_path (str): Path to the `.tfrecord` file that contains serialized text data
    """

    dataset_stage_1 = tf.data.Dataset.list_files("{}/*".format(input_path), shuffle=shuffle, seed=shuffle_seed)
    dataset_stage_1 = dataset_stage_1.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )    

    features_description = {
        'question': tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
        'positive_ctxs': tf.io.VarLenFeature(dtype=tf.string),
        'negative_ctxs': tf.io.VarLenFeature(dtype=tf.string),
        'hard_negative_ctxs': tf.io.VarLenFeature(dtype=tf.string)
    }
    def _parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, features_description)

    dataset_stage_1 = dataset_stage_1.map(
        _parse_example,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ctx_features_description = {
        'title': tf.io.FixedLenFeature([1], dtype=tf.string),
        'text': tf.io.FixedLenFeature([1], dtype=tf.string)
    }
    def _parse_ctx_example(ctx_example_proto):
        return tf.io.parse_single_example(ctx_example_proto, ctx_features_description)

    def _deserialize_ctxs():
        # count = 0
        for element in dataset_stage_1:
            question = element['question'][0]

            positive_ctxs   = element['positive_ctxs']
            positive_ctxs   = [_parse_ctx_example(ctx) for ctx in positive_ctxs.values]
            positive_titles = [ctx['title'][0] for ctx in positive_ctxs]
            positive_texts  = [ctx['text'][0] for ctx in positive_ctxs]
            positive_titles = tf.convert_to_tensor(positive_titles, dtype=tf.string)
            positive_texts  = tf.convert_to_tensor(positive_texts, dtype=tf.string)
            positive_titles = tf.sparse.from_dense(positive_titles)
            positive_texts  = tf.sparse.from_dense(positive_texts)
            
            negative_ctxs   = element['negative_ctxs']
            negative_ctxs   = [_parse_ctx_example(ctx) for ctx in negative_ctxs.values]
            negative_titles = [ctx['title'][0] for ctx in negative_ctxs]
            negative_texts  = [ctx['text'][0] for ctx in negative_ctxs]
            negative_titles = tf.convert_to_tensor(negative_titles, dtype=tf.string)
            negative_texts  = tf.convert_to_tensor(negative_texts, dtype=tf.string)
            negative_titles = tf.sparse.from_dense(negative_titles)
            negative_texts  = tf.sparse.from_dense(negative_texts)

            hard_negative_ctxs   = element['hard_negative_ctxs']
            hard_negative_ctxs   = [_parse_ctx_example(ctx) for ctx in hard_negative_ctxs.values]
            hard_negative_titles = [ctx['title'][0] for ctx in hard_negative_ctxs]
            hard_negative_texts  = [ctx['text'][0] for ctx in hard_negative_ctxs]
            hard_negative_titles = tf.convert_to_tensor(hard_negative_titles, dtype=tf.string)
            hard_negative_texts  = tf.convert_to_tensor(hard_negative_texts, dtype=tf.string)
            hard_negative_titles = tf.sparse.from_dense(hard_negative_titles)
            hard_negative_texts  = tf.sparse.from_dense(hard_negative_texts)

            # count += 1
            # print("Count: {}".format(count))

            yield {
                'question': question,
                'positive_ctxs': {
                    'title': positive_titles,
                    'text': positive_texts,
                },
                'negative_ctxs': {
                    'title': negative_titles,
                    'text': negative_texts
                },
                'hard_negative_ctxs': {
                    'title': hard_negative_titles,
                    'text': hard_negative_texts
                }
            }

    dataset_stage_2 = tf.data.Dataset.from_generator(
        _deserialize_ctxs,
        output_signature={
            'question': tf.TensorSpec([], tf.string),
            'positive_ctxs': {
                'title': tf.SparseTensorSpec([None], tf.string),
                'text': tf.SparseTensorSpec([None,], tf.string)
            },
            'negative_ctxs': {
                'title': tf.SparseTensorSpec([None], tf.string),
                'text': tf.SparseTensorSpec([None], tf.string)
            },
            'hard_negative_ctxs': {
                'title': tf.SparseTensorSpec([None], tf.string),
                'text': tf.SparseTensorSpec([None], tf.string)
            }
        }
    )

    return dataset_stage_2


def transform_retriever_data_from_text_to_int(
    dataset: tf.data.Dataset,
    tensorizer: Tensorizer
):
    def _transform():
        # count = 0
        for element in dataset:
            question = element['question']
            question_tensor = tensorizer.tensorize_single_nonpad(question.numpy().decode())

            positive_ctxs = element['positive_ctxs']
            positive_titles = positive_ctxs['title'].values.numpy()
            positive_titles = [title.decode(errors='ignore') for title in positive_titles]
            positive_texts  = positive_ctxs['text'].values.numpy()
            positive_texts  = [text.decode(errors='ignore') for text in positive_texts]

            assert len(positive_titles) == len(positive_texts)

            hard_negative_ctxs = element['hard_negative_ctxs']
            hard_negative_titles = hard_negative_ctxs['title'].values.numpy()
            hard_negative_titles = [title.decode(errors='ignore') for title in hard_negative_titles]
            hard_negative_texts = hard_negative_ctxs['text'].values.numpy()
            hard_negative_texts = [text.decode(errors='ignore') for text in hard_negative_texts]

            assert len(hard_negative_titles) == len(hard_negative_texts)

            # Merging
            positive_passages = [{
                'title': positive_title,
                'text': positive_text
            } for positive_title, positive_text in zip(positive_titles, positive_texts)]
            positive_tensor = tensorizer.tensorize_context_nonpad(
                positive_passages,
            )
            if positive_tensor is None:
                continue

            hard_negative_passages = [{
                'title': hard_negative_title,
                'text': hard_negative_text
            } for hard_negative_title, hard_negative_text in zip(hard_negative_titles, hard_negative_texts)]
            hard_negative_tensor = tensorizer.tensorize_context_nonpad(
                hard_negative_passages
            )
            if hard_negative_tensor is None:
                continue
            
            positive_scores = tf.ones([positive_tensor.shape[0]], dtype=tf.int32)
            negative_scores = tf.zeros([hard_negative_tensor.shape[0]], dtype=tf.int32)
            target_scores   = tf.concat([positive_scores, negative_scores], axis=0)

            # count += 1
            # print("Count: {}".format(count))

            yield {
                'question': tf.sparse.from_dense(question_tensor), # 1-D tensor
                'positive_tensor': positive_tensor.to_sparse(), # 2-D tensor
                'hard_negative_tensor': hard_negative_tensor.to_sparse(), # 2-D tensor
                'target_scores': tf.sparse.from_dense(target_scores) # 1-D tensor
            }
    
    return tf.data.Dataset.from_generator(
        _transform,
        output_signature={
            'question': tf.SparseTensorSpec(shape=[None], dtype=tf.int32),
            'positive_tensor': tf.SparseTensorSpec(shape=[None, None], dtype=tf.int32),
            'hard_negative_tensor': tf.SparseTensorSpec(shape=[None, None], dtype=tf.int32),
            'target_scores': tf.SparseTensorSpec(shape=[None], dtype=tf.int32)
        }
    )


def serialize_retriever_int_data(
    dataset: tf.data.Dataset
):
    def _serialize(record):
        question = tf.io.serialize_tensor(tf.io.serialize_sparse(record['question']))
        positive_tensor = tf.io.serialize_tensor(tf.io.serialize_sparse(record['positive_tensor']))
        hard_negative_tensor = tf.io.serialize_tensor(tf.io.serialize_sparse(record['hard_negative_tensor']))
        target_scores = tf.io.serialize_tensor(tf.io.serialize_sparse(record['target_scores']))

        features = {
            'question': tf.train.Feature(bytes_list=tf.train.BytesList(value=[question.numpy()])),
            'positive_tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[positive_tensor.numpy()])),
            'hard_negative_tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hard_negative_tensor.numpy()])),
            'target_scores': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_scores.numpy()]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    def _generate():
        count = 0
        for element in dataset:
            count += 1
            print("Count: {}".format(count))
            yield _serialize(element)

    return tf.data.Dataset.from_generator(
        _generate,
        output_signature=tf.TensorSpec([], tf.string)
    )


def load_retriever_tfrecord_int_data(
    data_path: str,
    shuffle: bool = True,
    shuffle_seed: int = 123
):
    dataset = tf.data.Dataset.list_files("{}/*".format(data_path), shuffle=shuffle, seed=shuffle_seed)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    features_description = {
        'question': tf.io.FixedLenFeature([1], tf.string),
        'positive_tensor': tf.io.FixedLenFeature([1], tf.string),
        'hard_negative_tensor': tf.io.FixedLenFeature([1], tf.string),
        'target_scores': tf.io.FixedLenFeature([1], tf.string)
    }

    def _deserialize(example_proto):
        record = tf.io.parse_single_example(example_proto, features_description)
        
        question_serialized = record['question'][0]
        question_sparse_serialized = tf.io.parse_tensor(question_serialized, out_type=tf.string)
        question_values = tf.io.parse_tensor(question_sparse_serialized[1], out_type=tf.int32)

        positive_serialized = record['positive_tensor'][0]
        positive_sparse_serialized = tf.io.parse_tensor(positive_serialized, out_type=tf.string)
        positive_indices = tf.io.parse_tensor(positive_sparse_serialized[0], out_type=tf.int64)
        positive_values = tf.io.parse_tensor(positive_sparse_serialized[1], out_type=tf.int32)
        positive_dense_shape = tf.io.parse_tensor(positive_sparse_serialized[2], out_type=tf.int64)
        positive_tensor = tf.sparse.SparseTensor(
            indices=positive_indices,
            values=positive_values,
            dense_shape=positive_dense_shape
        )

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

        target_scores_serialized = record['target_scores'][0]
        target_scores_sparse_serialized = tf.io.parse_tensor(target_scores_serialized, out_type=tf.string)
        target_scores_values = tf.io.parse_tensor(target_scores_sparse_serialized[1], out_type=tf.int32)

        return {
            'question': question_values,
            'positive_tensor': positive_tensor,
            'hard_negative_tensor': hard_negative_tensor,
            'target_scores': target_scores_values
        }

    dataset = dataset.map(
        _deserialize,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def random_sampling(
    dataset: tf.data.Dataset,
    samples: int = 50
):
    """Note: Length of `target_scores` must be less than `samples`. Perform after padding
    """
    def _map_fn(element):
        context_ids = element['context_ids']
        context_masks = element['context_masks']
        target_scores = element['target_scores']
        target_scores_squeeze = tf.squeeze(target_scores, axis=0)

        upper_bound = tf.shape(target_scores_squeeze)[0]
        lower_bound = tf.math.count_nonzero(target_scores_squeeze, dtype=tf.int32)

        positive_indices = tf.range(1, lower_bound, dtype=tf.int32)
        max_positives = samples - 10
        positive_indices = tf.concat([[0], tf.random.shuffle(positive_indices)], axis=0)[:max_positives]
        num_positives = tf.shape(positive_indices)[0]
        
        hard_negative_indices = tf.random.shuffle(tf.range(lower_bound, upper_bound, dtype=tf.int32))[:samples - num_positives]

        indices = tf.concat([positive_indices, hard_negative_indices], axis=0)
        indices = tf.expand_dims(indices, axis=1)

        updates = tf.ones(tf.shape(indices)[0], dtype=tf.bool)
        mask = tf.scatter_nd(indices=indices, updates=updates, shape=tf.shape(target_scores_squeeze))

        context_ids = context_ids[mask]
        context_masks = context_masks[mask]

        target_scores = target_scores[:, :samples]

        return {
            **element,
            'context_ids': context_ids,
            'context_masks': context_masks,
            'target_scores': target_scores
        }

    return dataset.map(
        _map_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )


def pad(
    dataset: tf.data.Dataset,
    sep_token_id: int = 102,
    max_context_length: int = 256,
    max_query_length: int = 256,
):
    def _map_fn(element):
        question = element['question']
        question = question[:max_query_length]
        question = tf.pad(question, [[0, max_query_length - tf.shape(question)[0]]])
        question = tf.expand_dims(question, axis=0)

        positive_tensor  = element['positive_tensor']
        positive_tensor  = tf.sparse.to_dense(positive_tensor)
        positive_tensor  = positive_tensor[:, :max_context_length] # truncate
        positive_tensor  = tf.pad(positive_tensor, [[0, 0], [0, max_context_length - tf.shape(positive_tensor)[1]]]) # padding
        positive_shape   = tf.shape(positive_tensor)
        positive_scatter = tf.scatter_nd(indices=[[positive_shape[1] - 1]], updates=[tf.ones(positive_shape[0], dtype=tf.bool)], shape=[positive_shape[1], positive_shape[0]])
        positive_mask    = tf.transpose(positive_scatter, perm=[1, 0])
        positive_tensor  = tf.where(positive_mask, sep_token_id, positive_tensor)

        hard_negative_tensor  = element['hard_negative_tensor']
        hard_negative_tensor  = tf.sparse.to_dense(hard_negative_tensor)
        hard_negative_tensor  = hard_negative_tensor[:, :max_context_length] # truncate
        hard_negative_tensor  = tf.pad(hard_negative_tensor, [[0, 0], [0, max_context_length - tf.shape(hard_negative_tensor)[1]]]) # padding
        hard_negative_shape   = tf.shape(hard_negative_tensor)
        hard_negative_scatter = tf.scatter_nd(indices=[[hard_negative_shape[1] - 1]], updates=[tf.ones(hard_negative_shape[0], dtype=tf.bool)], shape=[hard_negative_shape[1], hard_negative_shape[0]])
        hard_negative_mask    = tf.cast(tf.transpose(hard_negative_scatter, perm=[1, 0]), dtype=tf.bool)
        hard_negative_tensor  = tf.where(hard_negative_mask, sep_token_id, hard_negative_tensor)

        target_scores = element['target_scores']
        target_scores = tf.pad(target_scores, [[0, hard_negative_shape[0]]])
        target_scores = tf.expand_dims(target_scores, axis=0)

        question_mask = tf.cast(question > 0, tf.int32)
        context_tensor = tf.concat([positive_tensor, hard_negative_tensor], axis=0)
        context_mask = tf.cast(context_tensor > 0, tf.int32)

        return {
            'question_ids': question,
            'question_masks': question_mask,
            'context_ids': context_tensor,
            'context_masks': context_mask,
            'target_scores': target_scores
        }

    return dataset.map(
        _map_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default="data/retriever/V2/N5000-INT")
    args = parser.parse_args()

    input_path = args.input_path
    dataset = load_retriever_tfrecord_int_data(
        data_path=input_path
    )
    