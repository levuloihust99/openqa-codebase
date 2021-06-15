import os
import pickle
import time
import argparse
from functools import partial
from datetime import datetime

import tensorflow as tf
from transformers import BertConfig, TFBertModel
from bigbird.core import modeling

from dpr import const
from dpr.data import biencoder_manipulator
from dpr.models import get_encoder, get_tokenizer, get_model_input
from utilities import write_config, spread_samples_equally, spread_samples_greedy


def value_fn_template(ctx, indices, tensors):
    idxs = indices[ctx.replica_id_in_sync_group]
    return tensors[idxs[0] : idxs[1]]


def run(
    dataset,
    context_encoder,
    strategy,
    out_dir: str
):
    def eval_step(inputs):
        """The step function for one training step"""
        print("This function is tracing !")

        def step_fn(inputs):
            """The computation to be run on each compute device"""
            outputs = context_encoder(
                **inputs,
                training=False
            )

            seq_output, pooled_output = outputs[0], outputs[1]
            if not args.use_pooler:
                pooled_output = seq_output[:, 0, :]
            return pooled_output
            
        per_replica_outputs = strategy.run(step_fn, args=(inputs,))
        return per_replica_outputs

    if not args.disable_tf_function:
        eval_step = tf.function(eval_step)

    marked_time = time.perf_counter()
    begin = time.perf_counter()

    chunk = []
    index = 0
    count = 0
    iterator = iter(dataset)

    for element in iterator:
        if strategy.num_replicas_in_sync > 1:
            reduced_context_ids = tf.concat(element['context_ids'].values, axis=0)
        else:
            reduced_context_ids = element['context_ids']

        if strategy.num_replicas_in_sync > 1:
            global_passage_ids = tf.concat(element['passage_id'].values, axis=0)
        else:
            global_passage_ids = element['passage_id']
        global_passage_ids = [id.decode('utf-8') for id in global_passage_ids.numpy()]
        
        global_batch_size = reduced_context_ids.shape[0]

        if global_batch_size < args.batch_size * strategy.num_replicas_in_sync: # the last batch
            if strategy.num_replicas_in_sync > 1:
                reduced_context_masks = tf.concat(element['context_masks'].values, axis=0)
            else:
                reduced_context_masks = element['context_masks']

            global_batch_outputs = None
            base_replica_batch_size = args.batch_size

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
                    
                    value_fn_for_context_ids = partial(value_fn_template, indices=indices, tensors=reduced_context_ids)
                    value_fn_for_context_masks = partial(value_fn_template, indices=indices, tensors=reduced_context_masks)

                    reduced_context_ids = reduced_context_ids[base_replica_batch_size * strategy.num_replicas_in_sync:]
                    reduced_context_masks = reduced_context_masks[base_replica_batch_size * strategy.num_replicas_in_sync:]

                    dist_context_ids = strategy.experimental_distribute_values_from_function(value_fn_for_context_ids)
                    dist_context_masks = strategy.experimental_distribute_values_from_function(value_fn_for_context_masks)

                    element = get_model_input(
                        input_ids=dist_context_ids,
                        atttention_mask=dist_context_masks,
                        model_name=args.pretrained_model
                    )

                    per_replica_outputs = eval_step(element)
                    if global_batch_outputs is None:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = tf.concat(per_replica_outputs.values, axis=0)
                        else:
                            global_batch_outputs = per_replica_outputs
                    else:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = tf.concat([global_batch_outputs, *per_replica_outputs.values], axis=0)
                        else:
                            global_batch_outputs = tf.concat([global_batch_outputs, per_replica_outputs], axis=0)

                    if global_batch_size == 0:
                        break

                else:
                    element = get_model_input(
                        input_ids=reduced_context_ids,
                        attention_mask=reduced_context_masks,
                        model_name=args.pretrained_model
                    )

                    per_replica_outputs = eval_step(element)

                    if global_batch_outputs is None:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = per_replica_outputs.values[0]
                        else:
                            global_batch_outputs = per_replica_outputs
                    else:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = tf.concat([global_batch_outputs, per_replica_outputs.values[0]], axis=0)
                        else:
                            global_batch_outputs = tf.concat([global_batch_outputs, per_replica_outputs], axis=0)

                    break

        else:
            element = get_model_input(
                input_ids=element['context_ids'],
                attention_mask=element['context_masks'],
                model_name=args.pretrained_model
            )

            per_replica_outputs = eval_step(element) # Run on TPU

            if strategy.num_replicas_in_sync > 1:
                global_batch_outputs = tf.concat(per_replica_outputs.values, axis=0)
            else:
                global_batch_outputs = per_replica_outputs

        global_batch = list(zip(global_passage_ids, global_batch_outputs))

        count += 1
        print("Step: {: <10} - Elapsed: {}".format(count, time.perf_counter() - begin))
        begin = time.perf_counter()

        if len(chunk) + len(global_batch) >= args.records_per_file:
            delta = len(chunk) + len(global_batch) - args.records_per_file
            if delta > 0:
                kept_part = global_batch[:-delta]
                remain_part = global_batch[-delta:]
            else:
                kept_part = global_batch
                remain_part = []

            chunk.extend(kept_part)
            with open(os.path.join(out_dir, "wikipedia_passages_{}.pkl".format(index)), "wb") as writer:
                pickle.dump(chunk, writer)
                print("{: <30}{}".format(time.perf_counter() - marked_time, os.path.abspath(writer.name)))
            marked_time = time.perf_counter()
            
            chunk = remain_part
            index += 1
        
        else:
            chunk.extend(global_batch)
        
    if chunk:
        with open(os.path.join(out_dir, "wikipedia_passages_{:02d}.pkl".format(index)), "wb") as writer:
            pickle.dump(chunk, writer)
            print("{: <30}{}".format(time.perf_counter() - marked_time, os.path.abspath(writer.name)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=const.CHECKPOINT_PATH)
    parser.add_argument("--ctx-source-shards-tfrecord", type=str, default=const.CTX_SOURCE_SHARDS_TFRECORD)
    parser.add_argument("--records-per-file", type=int, default=const.RECORDS_PER_FILE)
    parser.add_argument("--embeddings-path", type=str, default=const.EMBEDDINGS_DIR)
    parser.add_argument("--seed", type=int, default=const.SHUFFLE_SEED)
    parser.add_argument("--batch-size", type=int, default=const.EVAL_BATCH_SIZE)
    parser.add_argument("--tpu", type=str, default=const.TPU_NAME)
    parser.add_argument("--max-context-length", type=int, default=const.MAX_CONTEXT_LENGTH, help="Maximum length of a document")
    parser.add_argument("--pretrained-model", type=str, default=const.PRETRAINED_MODEL)
    parser.add_argument("--use-pooler", type=eval, default=True)
    parser.add_argument("--disable-tf-function", type=eval, default=False)
    parser.add_argument("--prefix", type=str, default='pretraineds')

    global args
    args = parser.parse_args()
    model_type = os.path.basename(args.checkpoint_path)
    embeddings_path = os.path.join(args.embeddings_path, "shards-42031", model_type)
    args_dict = {**args.__dict__, "embeddings_path": embeddings_path}

    configs = ["{}: {}".format(k, v) for k, v in args_dict.items()]
    configs_string = "\t" + "\n\t".join(configs) + "\n"
    print("************************* Configurations *************************")
    print(configs_string)
    print("----------------------------------------------------------------------------------------------------------------------")

    config_path = "configs/{}/{}/config.yml".format(__file__.rstrip(".py"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
            strategy = tf.distribute.MirroredStrategy(["GPU:0"])
        else:
            strategy = tf.distribute.get_strategy()

    tf.random.set_seed(args.seed)

    """
    Data pipeline
    """
    print("Data pipeline processing...")
    dataset = biencoder_manipulator.load_tfrecord_tokenized_data_for_ctx_sources(
        input_path=args.ctx_source_shards_tfrecord,
        max_context_length=args.max_context_length
    )
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Distribute the dataset
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )

    print("done")
    print("----------------------------------------------------------------------------------------------------------------------")

    """
    Load checkpoint
    """
    print("Loading checkpoint...")
    checkpoint_path = args.checkpoint_path
    with strategy.scope():
        context_encoder = get_encoder(
            model_name=args.pretrained_model,
            args=args,
            trainable=False,
            prefix=args.prefix
        )

        retriever = tf.train.Checkpoint(ctx_model=context_encoder)
        root_ckpt = tf.train.Checkpoint(model=retriever)

        root_ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    
    print(tf.train.latest_checkpoint(checkpoint_path))

    print("done")
    print("----------------------------------------------------------------------------------------------------------------------")

    """
    Generate embeddings
    """
    print("Generate embeddings...")
    out_dir = embeddings_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run(
        dataset=dist_dataset,
        strategy=strategy,
        context_encoder=context_encoder,
        out_dir=out_dir
    )

    print("done")
    print("----------------------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()