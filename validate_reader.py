import argparse
from dpr.data import reader_manipulator
import os
from datetime import datetime
import time
from multiprocessing import Pool as ProcessPool
from functools import partial
from tqdm import tqdm

import tensorflow as tf
from transformers import TFBertModel, BertConfig, BertTokenizer

from dpr.utils.span_validation import get_best_span, compare_spans
from dpr import models


def validate(
    reader,
    strategy,
    dataset,
    params=None
):
    print("Validating...")

    if params is not None:
        global args
        args = params
    
    def dist_forward_step(input_ids):
        print("This function is tracing")

        def step_fn(input_ids):
            attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)

            start_logits, end_logits = reader(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )

            return start_logits, end_logits

        per_replica_logits = strategy.run(step_fn, args=(input_ids,))
        return per_replica_logits
    
    if not args.disable_tf_function:
        dist_forward_step = tf.function(dist_forward_step)

    processes = ProcessPool(processes=os.cpu_count())
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    get_best_span_partial = partial(get_best_span, max_answers=args.max_answers, tokenizer=tokenizer)
        
    iterator = iter(dataset)
    em_hits = []
    match_stats = []
    for element in tqdm(iterator):
        answers_serialized = element['answers']
        question = element['question']
        passage_offset = element['passages/passage_offset']
        input_ids = element['input_ids']

        start_logits, end_logits = dist_forward_step(input_ids)
        if strategy.num_replicas_in_sync > 1:
            start_logits = tf.concat(start_logits.values, axis=0)
            end_logits = tf.concat(end_logits.values, axis=0)
            input_ids = tf.concat(input_ids, axis=0)
            passage_offset = tf.concat(passage_offset, axis=0)

        sentence_ids = tf.RaggedTensor.from_tensor(input_ids, padding=tokenizer.pad_token_id)
        sentence_ids = sentence_ids.to_list()
        ctx_ids = [ids[offset:] for ids, offset in zip(sentence_ids, passage_offset)]

        start_logits = start_logits.numpy().tolist()
        start_logits = [logits[offset : offset + len(ctx)] for logits, offset, ctx in zip(start_logits, passage_offset, ctx_ids)]
        end_logits = end_logits.numpy().tolist()
        end_logits = [logits[offset : offset + len(ctx)] for logits, offset, ctx in zip(end_logits, passage_offset, ctx_ids)]
        
        best_spans = processes.starmap(get_best_span_partial, zip(start_logits, end_logits, ctx_ids))

        answers = []
        for ans in answers_serialized:
            ans_sparse = tf.io.parse_tensor(ans, out_type=tf.string)
            ans_values = tf.io.parse_tensor(ans_sparse[1], out_type=tf.string)
            ans_values = [answer.numpy().decode() for answer in ans_values]
            answers.append(ans_values)

        hits = processes.startmap(compare_spans, zip(answers, best_spans))
        passages = [tokenizer.decode(ids) for ids in ctx_ids]

        stats = [
            {
                "question": q,
                "answers": ans,
                "passage": psg,
                "predicted": span,
                "hit": hit
            }
            for q, ans, span, psg, hit in zip(question, answers, best_spans, passages, hits)
        ]
        match_stats.extend(stats)
        em_hits.extend(hits)

    print("done")
    print("-----------------------------------------------------------")
    return em_hits, match_stats


def load_dataset(data_path: str, strategy):
    print("Prepare dataset for reader validation...")
    # Load dataset
    dataset = reader_manipulator.load_tfrecord_reader_dev_data(
        input_path=data_path
    )
    dataset = reader_manipulator.transform_to_reader_validate_dataset(
        dataset=dataset,
        max_sequence_length=args.max_sequence_length
    )
    dataset = dataset.batch(args.batch_size)
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
    checkpoint_path: str,
    strategy
):
    print("Loading checkpoint... ")

    config = BertConfig.from_pretrained(
        pretrained_model,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True
    )

    with strategy.scope():
        encoder = TFBertModel.from_pretrained(
            pretrained_model,
            config=config,
            trainable=False
        )

        reader = models.Reader(
            encoder=encoder,
            initializer_range=config.initializer_range
        )

        ckpt = tf.train.Checkpoint(model=reader)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

        print("Checkpoint file: {}".format(tf.train.latest_checkpoint(checkpoint_path)))
        print("done")
        print("-----------------------------------------------------------")

    return reader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="gs://openqa-dpr/data/reader/nq/dev/dev.tfrecord")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--tpu", type=str, default="tpu-v3")
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-path", type=str, default="gs://openqa-dpr/checkpoints/reader/baseline")

    global args
    args = parser.parse_args()

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

    dataset = load_dataset(data_path=args.data_path, strategy=strategy)
    reader = load_checkpoint(pretrained_model=args.pretrained_model)

    validate(
        reader=reader,
        strategy=strategy,
        dataset=dataset
    )


if __name__ == "__main__":
    main()