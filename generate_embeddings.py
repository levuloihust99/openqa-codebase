import os
import pickle
import time
import argparse

import tensorflow as tf
from transformers import BertConfig, TFBertModel

from dpr import const
from dpr.data import manipulator


def run(
    dataset,
    context_encoder,
    strategy,
    out_dir: str
):
    @tf.function
    def eval_step(element):
        """The step function for one training step"""
        print("This function is tracing !")

        def step_fn(element):
            """The computation to be run on each compute device"""
            context_ids = element['context_ids']
            context_masks = element['context_masks']

            outputs = context_encoder(
                input_ids=context_ids,
                attention_mask=context_masks,
                training=False
            )

            seq_output, pooled_output = outputs[0], outputs[1]
            return pooled_output
            
        per_replica_outputs = strategy.run(step_fn, args=(element,))
        return per_replica_outputs

    marked_time = time.perf_counter()
    begin = time.perf_counter()

    chunk = []
    idx = 0
    count = 0
    iterator = iter(dataset)

    for element in iterator:
        per_replica_outputs = eval_step(element) # Run on TPU

        # Run on CPU
        if strategy.num_replicas_in_sync > 1:
            global_batch_outputs = tf.concat(per_replica_outputs.values, axis=0)
            global_passage_ids = tf.concat(element['passage_id'].values, axis=0)
        else:
            global_batch_outputs = per_replica_outputs
            global_passage_ids = element['passage_id']
        
        global_passage_ids = [id.decode('utf-8') for id in global_passage_ids.numpy()]
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
            with open(os.path.join(out_dir, "wikipedia_passages_{}.pkl".format(idx)), "wb") as writer:
                pickle.dump(chunk, writer)
                print("{: <30}{}".format(time.perf_counter() - marked_time, os.path.abspath(writer.name)))
            marked_time = time.perf_counter()
            
            chunk = remain_part
            idx += 1
        
        else:
            chunk.extend(global_batch)
        
    if chunk:
        with open(os.path.join(out_dir, "wikipedia_passages_{:02d}.pkl".format(idx)), "wb") as writer:
            pickle.dump(chunk, writer)
            print("{: <30}{}".format(time.perf_counter() - marked_time, os.path.abspath(writer.name)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=const.CHECKPOINT_PATH)
    parser.add_argument("--ctx-source-shards-tfrecord", type=str, default=const.CTX_SOURCE_SHARDS_TFRECORD)
    parser.add_argument("--records-per-file", type=int, default=42031)
    parser.add_argument("--embeddings-dir", type=str, default=const.EMBEDDINGS_DIR)
    parser.add_argument("--seed", type=int, default=const.SHUFFLE_SEED)
    parser.add_argument("--batch-size", type=int, default=const.EVAL_BATCH_SIZE)
    parser.add_argument("--tpu", type=str, default=const.TPU_NAME)
    parser.add_argument("--gcloud-bucket", type=str, default=const.STORAGE_BUCKET)
    parser.add_argument("--model-type", type=str, default=const.DEFAULT_MODEL)
    parser.add_argument("--max-context-length", type=int, default=const.MAX_CONTEXT_LENGTH, help="Maximum length of a document")

    global args
    args = parser.parse_args()

    print("******************* Configurations *******************\n")
    configs = ["{}: {}".format(k, v) for k, v in args.__dict__.items()]
    configs_string = "\t" + "\n\t".join(configs) + "\n"
    print(configs_string)

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
    dataset = manipulator.load_tfrecord_tokenized_data_for_ctx_sources(
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
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True
    )
    checkpoint_path = args.checkpoint_path

    with strategy.scope():
        context_encoder = TFBertModel.from_pretrained(
            'bert-base-uncased',
            config=config,
            trainable=False
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
    out_dir = os.path.join(args.embeddings_dir, args.model_type, "shards-{}".format(args.records_per_file))
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