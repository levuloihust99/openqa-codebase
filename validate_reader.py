import argparse
from dpr.data import reader_manipulator
import os
from datetime import datetime
import time
from multiprocessing import Pool as ProcessPool
from functools import partial
from tqdm import tqdm
import json

import tensorflow as tf
from transformers import TFBertModel, BertConfig, BertTokenizer
import numpy as np

from dpr.utils.span_validation import get_best_span, compare_spans
from dpr import models
from utilities import write_config, spread_samples_equally


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

    def value_fn_template(ctx, indices, tensors):
        start, end = indices[ctx.replica_id_in_sync_group]
        return tensors[start : end]

    processes = ProcessPool(processes=os.cpu_count())
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    get_best_span_partial = partial(get_best_span, max_answer_length=args.max_answer_length, tokenizer=tokenizer)
        
    iterator = iter(dataset)
    em_hits = []
    match_stats = []
    for element in tqdm(iterator):
        answers_serialized = element['answers']
        question = element['question']
        passage_offset = element['passage_offset']
        input_ids = element['input_ids']

        if strategy.num_replicas_in_sync > 1:
            reduced_input_ids = tf.concat(input_ids.values, axis=0)
        else:
            reduced_input_ids = input_ids
        
        global_batch_size = reduced_input_ids.shape[0]
        
        # forward pass
        if global_batch_size < args.batch_size * strategy.num_replicas_in_sync:
            base_replica_batch_size = args.batch_size
            flag = False
        
            while True:
                spread, global_batch_size, base_replica_batch_size = spread_samples_equally(
                    global_batch_size=global_batch_size,
                    num_replicas=strategy.num_replicas_in_sync,
                    base_replica_batch_size=base_replica_batch_size,
                    init_batch_size=args.batch_size
                )

                if len(spread) > 1:
                    indices = []
                    idx = 0
                    for num in spread:
                        indices.append((idx, idx + num))
                        idx += num
                    
                    value_fn = partial(value_fn_template, indices=indices, tensors=reduced_input_ids)
                    reduced_input_ids = reduced_input_ids[base_replica_batch_size * strategy.num_replicas_in_sync:]
                    dist_input_ids = strategy.experimental_distribute_values_from_function(value_fn)
                    start_logits, end_logits = dist_forward_step(dist_input_ids)
                    
                    if not flag:
                        if strategy.num_replicas_in_sync > 1:
                            global_start_logits = tf.concat(start_logits.values, axis=0)
                            global_end_logits = tf.concat(end_logits.values, axis=0)
                        else:
                            global_start_logits = start_logits
                            global_end_logits = end_logits
                        flag = True

                    else:
                        if strategy.num_replicas_in_sync > 1:
                            global_start_logits = tf.concat([global_start_logits, *start_logits.values], axis=0)
                            global_end_logits = tf.concat([global_end_logits, *start_logits.values], axis=0)
                        else:
                            global_start_logits = tf.concat([global_start_logits, start_logits], axis=0)
                            global_end_logits = tf.concat([global_end_logits, end_logits], axis=0)

                    if global_batch_size == 0:
                        break

                else:
                    start_logits, end_logits = dist_forward_step(reduced_input_ids)
                    if not flag:
                        if strategy.num_replicas_in_sync > 1:
                            global_start_logits = start_logits.values[0]
                            global_end_logits = end_logits.values[0]
                        else:
                            global_start_logits = start_logits
                            global_end_logits = end_logits
                        flag = True
                    else:
                        if strategy.num_replicas_in_sync > 1:
                            global_start_logits = tf.concat([global_start_logits, start_logits.values[0]], axis=0)
                            global_end_logits = tf.concat([global_end_logits, end_logits.values[0]], axis=0)
                        else:
                            global_start_logits = tf.concat([global_start_logits, start_logits], axis=0)
                            global_end_logits = tf.concat([global_end_logits, end_logits], axis=0)
                    break
 
        else:
            start_logits, end_logits = dist_forward_step(input_ids)
            if strategy.num_replicas_in_sync > 1:
                global_start_logits = tf.concat(start_logits.values, axis=0)
                global_end_logits = tf.concat(end_logits.values, axis=0)
            else:
                global_start_logits = start_logits
                global_end_logits = end_logits
        
        if strategy.num_replicas_in_sync > 1:
            input_ids = tf.concat(input_ids.values, axis=0)
            passage_offset = tf.concat(passage_offset.values, axis=0)
            answers_serialized = tf.concat(answers_serialized.values, axis=0)
            question = tf.concat(question.values, axis=0)
        
        question = question.numpy().tolist()
        question = [q.decode() for q in question]
        
        sentence_ids = tf.RaggedTensor.from_tensor(input_ids, padding=tokenizer.pad_token_id)
        sentence_ids = sentence_ids.to_list()
        ctx_ids = [ids[offset:] for ids, offset in zip(sentence_ids, passage_offset)]

        start_logits = global_start_logits.numpy().tolist()
        start_logits = [logits[offset : offset + len(ctx)] for logits, offset, ctx in zip(start_logits, passage_offset, ctx_ids)]
        end_logits = global_end_logits.numpy().tolist()
        end_logits = [logits[offset : offset + len(ctx)] for logits, offset, ctx in zip(end_logits, passage_offset, ctx_ids)]
        
        best_spans = processes.starmap(get_best_span_partial, zip(start_logits, end_logits, ctx_ids))

        answers = []
        for ans in answers_serialized:
            ans_sparse = tf.io.parse_tensor(ans, out_type=tf.string)
            ans_values = tf.io.parse_tensor(ans_sparse[1], out_type=tf.string)
            ans_values = [answer.numpy().decode() for answer in ans_values]
            answers.append(ans_values)

        hits = processes.starmap(compare_spans, zip(answers, best_spans))
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
        pretrained_model_path,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True
    )

    with strategy.scope():
        encoder = TFBertModel.from_pretrained(
            pretrained_model_path,
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

    parser.add_argument("--data-path", type=str, default="gs://openqa-dpr/data/reader/nq/dev/dev.tfrecord")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--tpu", type=str, default="tpu-v3")
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-path", type=str, default="gs://openqa-dpr/checkpoints/reader/baseline")
    parser.add_argument("--disable-tf-function", type=eval, default=False)
    parser.add_argument("--max-answer-length", type=int, default=10)
    parser.add_argument("--res-dir", type=str, default="results/reader")
    parser.add_argument("--pretrained-model", type=str, default='bert-base-uncased')
    parser.add_argument("--prefix", type=str, default='pretrained')

    global args
    args = parser.parse_args()
    args_dict = args.__dict__
    checkpoint_type = os.path.basename(args.checkpoint_path)
    args_dict['res_dir'] = os.path.join(args_dict['res_dir'], checkpoint_type)

    configs = ["{}: {}".format(k, v) for k, v in args_dict.items()]
    configs_string = "\t" + "\n\t".join(configs) + "\n"
    print("************************* Configurations *************************")
    print(configs_string)
    print("----------------------------------------------------------------------------------------------------------------------")

    config_path = "configs/{}/{}/config.yml".format(os.path.basename(__file__).rstrip(".py"), datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    write_config(config_path, args_dict)

    if 'prefix' in args:
        global pretrained_model_path
        pretrained_model_path = os.path.join(args.prefix, args.pretrained_model)

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
    reader = load_checkpoint(
        pretrained_model=pretrained_model_path,
        checkpoint_path=args.checkpoint_path,
        strategy=strategy
    )

    em_hits, match_stats = validate(
        reader=reader,
        strategy=strategy,
        dataset=dataset
    )

    save_results(em_hits=em_hits, match_stats=match_stats, out_dir=args.res_dir)


if __name__ == "__main__":
    main()