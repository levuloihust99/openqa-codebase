import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime
import time
from functools import partial
from multiprocessing import Pool as ProcessPool

import tensorflow as tf
from transformers import BertConfig, TFBertModel, BertTokenizer
import numpy as np

from dpr.utils.span_validation import get_best_span, compare_spans
from dpr.data import reader_manipulator
from dpr import models
from dpr.models import get_encoder, get_tokenizer, get_config
from utilities import write_config, spread_samples_equally


def validate(
    dataset,
    strategy,
    ranker,
    reader,
    params=None
):
    if params is not None:
        global args
        args = params

    def dist_forward_ranker_step(input_ids):
        print("This function is tracing")

        def step_fn(input_ids):
            input_ids = tf.reshape(input_ids, [-1, args.max_sequence_length])
            attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)

            rank_logits = ranker(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )

            return tf.reshape(rank_logits, [args.batch_size, -1])

        per_replica_logits = strategy.run(step_fn, args=(input_ids,))
        return per_replica_logits

    def dist_forward_reader_step(input_ids):
        print("This function is tracing")

        def step_fn(input_ids):
            attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)

            start_logits, end_logits = reader(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )

            return start_logits, end_logits

        per_replica_results = strategy.run(step_fn, args=(input_ids,))
        return per_replica_results
    
    if not args.disable_tf_function:
        dist_forward_ranker_step = tf.function(dist_forward_ranker_step)
        dist_forward_reader_step = tf.function(dist_forward_reader_step)

    def value_fn_template(ctx, pool_tensors):
        return pool_tensors[ctx.replica_id_in_sync_group]
    
    processes = ProcessPool(processes=os.cpu_count())
    tokenizer = get_tokenizer(model_name=args.pretrained_model, prefix=args.prefix)

    get_best_span_partial = partial(get_best_span, max_answer_length=args.max_answer_length, tokenizer=tokenizer)

    iterator = iter(dataset)
    em_hits = []
    match_stats = []
    for element in tqdm(iterator):
        answers_serialized = element['answers']
        question = element['question']
        input_ids = element['passages/sequence_ids'] # bsz x num_passages x max_sequence_length
        passage_offsets = element['passages/passage_offset'] # bsz x num_passages

        reduced_input_ids = tf.concat(input_ids.values, axis=0)
        per_replica_passage_offsets = strategy.experimental_local_results(passage_offsets)
        
        global_batch_size = reduced_input_ids.shape[0]
        if global_batch_size < args.batch_size * strategy.num_replicas_in_sync:
            # TODO: add code in case batch is not divisible
            aggregated_input_ids = tf.concat(input_ids.values, axis=0)
            padded_size = args.batch_size * strategy.num_replicas_in_sync - global_batch_size
            padded_input_ids = tf.zeros([padded_size, args.max_passages, args.max_sequence_length], dtype=tf.int32)
            input_ids = tf.concat([aggregated_input_ids, padded_input_ids], axis=0)
            pool_input_ids = tf.split(input_ids, num_or_size_splits=strategy.num_replicas_in_sync, axis=0)
            value_fn_for_input_ids = partial(value_fn_template, pool_tensors=pool_input_ids)
            input_ids = strategy.experimental_distribute_values_from_function(value_fn_for_input_ids)

            aggregated_per_replica_passage_offsets = tf.concat(per_replica_passage_offsets, axis=0)
            lack_size = args.batch_size * strategy.num_replicas_in_sync - aggregated_per_replica_passage_offsets.shape[0]
            padded_per_replica_passage_offsets = tf.zeros([lack_size, args.max_passages], dtype=tf.int32)
            per_replica_passage_offsets = tf.concat([aggregated_per_replica_passage_offsets, padded_per_replica_passage_offsets], axis=0)
            per_replica_passage_offsets = tf.split(per_replica_passage_offsets, num_or_size_splits=strategy.num_replicas_in_sync)
        
        rank_logits = dist_forward_ranker_step(input_ids)
        rank_logits = strategy.experimental_local_results(rank_logits)
        selected_passage_idxs = [tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32) for logits in rank_logits] # num_replicas x batch_size
        
        selected_passage_offsets = []
        per_replica_input_ids = strategy.experimental_local_results(input_ids) # num_replicas x batch_sizse x max_passages x max_sequence_length
        selected_input_ids = []

        for sequence_ids, psg_offsets, passage_idxs  in zip(per_replica_input_ids, per_replica_passage_offsets, selected_passage_idxs):
            range_idxs = tf.range(sequence_ids.shape[0], dtype=tf.int32)
            indices = tf.concat(
                [
                    tf.expand_dims(range_idxs, axis=1),
                    tf.expand_dims(passage_idxs, axis=1)
                ],
                axis=1
            )
            selected_passage_offsets.append(
                tf.gather_nd(psg_offsets, indices)
            )
            selected_input_ids.append(
                tf.gather_nd(sequence_ids, indices)
            )
        
        value_fn = partial(value_fn_template, pool_tensors=selected_input_ids)
        dist_selected_input_ids = strategy.experimental_distribute_values_from_function(value_fn)
        
        start_logits, end_logits = dist_forward_reader_step(input_ids=dist_selected_input_ids)
        sentence_ids = tf.concat(dist_selected_input_ids.values, axis=0)
        sentence_ids = tf.RaggedTensor.from_tensor(sentence_ids, padding=tokenizer.pad_token_id)         
        sentence_ids = sentence_ids.to_list()
        sentence_ids = sentence_ids[:global_batch_size]
        selected_passage_offsets = tf.concat(selected_passage_offsets, axis=0)
        selected_passage_offsets = selected_passage_offsets[:global_batch_size]
        ctx_ids = [ids[offset:] for ids, offset in zip(sentence_ids, selected_passage_offsets)]

        start_logits = tf.concat(start_logits.values, axis=0)
        start_logits = start_logits.numpy().tolist()
        start_logits = start_logits[:global_batch_size]
        start_logits = [logits[offset : offset + len(ctx)] for logits, offset, ctx in zip(start_logits, selected_passage_offsets, ctx_ids)]
        end_logits = tf.concat(end_logits.values, axis=0)
        end_logits = end_logits.numpy().tolist()
        end_logits = end_logits[:global_batch_size]
        end_logits = [logits[offset : offset + len(ctx)] for logits, offset, ctx in zip(end_logits, selected_passage_offsets, ctx_ids)]

        best_spans = processes.starmap(get_best_span_partial, zip(start_logits, end_logits, ctx_ids))

        answers_serialized = tf.concat(answers_serialized.values, axis=0)
        question = tf.concat(question.values, axis=0)

        answers = []
        for ans in answers_serialized:
            ans_sparse = tf.io.parse_tensor(ans, out_type=tf.string)
            ans_values = tf.io.parse_tensor(ans_sparse[1], out_type=tf.string)
            ans_values = [answer.numpy().decode() for answer in ans_values]
            answers.append(ans_values)

        question = question.numpy().tolist()
        question = [q.decode() for q in question]

        hits = processes.starmap(compare_spans, zip(answers, best_spans))
        passages = [tokenizer.decode(ids) for ids in ctx_ids]

        selected_passage_idxs = tf.concat(selected_passage_idxs, axis=0)
        selected_passage_idxs = selected_passage_idxs.numpy().tolist()

        stats = [
            {
                "question": q,
                "answers": ans,
                "passage": psg,
                "predicted": span,
                "retriever_rank": idx + 1,
                "hit": hit
            }
            for q, ans, span, idx, psg, hit in zip(question, answers, best_spans, selected_passage_idxs, passages, hits)
        ]
        match_stats.extend(stats)
        em_hits.extend(hits)

    print("done")
    print("-----------------------------------------------------------")
    return em_hits, match_stats


def load_dataset(
    data_path: str,
    strategy,
    max_sequence_length: int = 256,
    max_passages: int = 50,
):
    print("Prepare dataset for reader validation...")
    # Load dataset
    dataset = reader_manipulator.load_tfrecord_reader_dev_data(
        input_path=data_path
    )
    dataset = reader_manipulator.transform_to_endtoend_validate_dataset(
        dataset=dataset,
        max_sequence_length=max_sequence_length,
        max_passages=max_passages
    )
    dataset = dataset.batch(args.batch_size)
    
    def _reshape(element):
        return {
            "answers": element['answers'],
            "question": element['question'],
            "passages/sequence_ids": tf.reshape(element['passages/sequence_ids'], [-1, args.max_passages, args.max_sequence_length]),
            "passages/passage_offset": tf.reshape(element['passages/passage_offset'], [-1, args.max_passages])
        }
    
    dataset = dataset.map(
        _reshape,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # distribute dataset
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )

    print("done")
    print("-----------------------------------------------------------")
    return dist_dataset


def load_checkpoint(
    pretrained_model: str,
    ranker_checkpoint_path: str,
    reader_checkpoint_path: str,
    strategy
):
    print("Loading checkpoint... ")

    config = get_config(
        model_name=args.pretrained_model,
        prefix=args.prefix
    )
    with strategy.scope():
        encoder = get_encoder(
            model_name=pretrained_model,
            args=args,
            trainable=False,
            prefix=args.prefix
        )

        ranker = models.Ranker(
            encoder=encoder,
            initializer_range=config.initializer_range,
            use_pooler=args.use_pooler,
            trainable=False
        )

        reader = models.Reader(
            encoder=encoder,
            initializer_range=config.initializer_range,
            trainable=False
        )

        ranker_ckpt = tf.train.Checkpoint(model=ranker)
        ranker_ckpt.restore(tf.train.latest_checkpoint(ranker_checkpoint_path)).expect_partial()
        reader_ckpt = tf.train.Checkpoint(model=reader)
        reader_ckpt.restore(tf.train.latest_checkpoint(reader_checkpoint_path)).expect_partial()

    print("Ranker checkpoint file: {}".format(tf.train.latest_checkpoint(ranker_checkpoint_path)))
    print("Reader checkpoint file: {}".format(tf.train.latest_checkpoint(reader_checkpoint_path)))
    print("done")
    print("-----------------------------------------------------------")

    return ranker, reader


def save_results(em_hits, match_stats, out_dir):
    print("Saving results...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    em_score = np.mean(em_hits)
    with open(os.path.join(out_dir, "em_score.txt"), "w") as writer:
        writer.write(str(em_score))
    with open(os.path.join(out_dir, "match_stats.json"), "w") as writer:
        json.dump(match_stats, writer, indent=4)

    print("done")
    print("-----------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="gs://openqa-dpr/data/reader/nq/dev/dev_tiny.tfrecord")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--max-passages", type=int, default=50)
    parser.add_argument("--max-answer-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tpu", type=str, default="tpu-v3-sanji")
    parser.add_argument("--ranker-checkpoint-path", type=str, default="gs://openqa-dpr/checkpoints/ranker/baseline")
    parser.add_argument("--reader-checkpoint-path", type=str, default="gs://openqa-dpr/checkpoints/reader/baseline")
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")
    parser.add_argument("--use-pooler", type=eval, default=False)
    parser.add_argument("--disable-tf-function", type=eval, default=False)
    parser.add_argument("--res-dir", type=str, default="results/endtoend/baseline")
    parser.add_argument("--prefix", type=str, default='pretrained')

    global args
    args = parser.parse_args()
    args_dict = args.__dict__

    configs = ["{}: {}".format(k, v) for k, v in args_dict.items()]
    configs_string = "\t" + "\n\t".join(configs) + "\n"
    print("************************* Configurations *************************")
    print(configs_string)
    print("----------------------------------------------------------------------------------------------------------------------")

    config_path = "configs/{}/{}/config.yml".format(os.path.basename(__file__).rstrip(".py"), datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    write_config(config_path, args_dict)

    try: # detect TPUs
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu) # TPU detection
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except Exception: # detect GPUs
        devices = tf.config.list_physical_devices("GPU")
        # [tf.config.experimental.set_memory_growth(device, True) for device in devices]
        if devices:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

    dataset = load_dataset(
        data_path=args.data_path,
        strategy=strategy,
        max_sequence_length=args.max_sequence_length,
        max_passages=args.max_passages
    )

    ranker, reader = load_checkpoint(
        pretrained_model=args.pretrained_model,
        ranker_checkpoint_path=args.ranker_checkpoint_path,
        reader_checkpoint_path=args.reader_checkpoint_path,
        strategy=strategy
    )

    em_hits, match_stats = validate(
        dataset=dataset,
        strategy=strategy,
        ranker=ranker,
        reader=reader,
    )

    save_results(
        em_hits=em_hits,
        match_stats=match_stats,
        out_dir=args.res_dir
    )

    print("done")

if __name__ == "__main__":
    main()