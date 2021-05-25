import tensorflow as tf

from transformers import TFBertModel, BertTokenizer, BertConfig

import os
import argparse
import time
from tqdm import tqdm

from dpr import const, models, losses, optimizers
from dpr.data import manipulator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data-size", type=int, default=const.TRAIN_DATA_SIZE)
    parser.add_argument("--data-path", type=str, default=const.DATA_PATH, help="Path to the `.tfrecord` data. Data in this file is already preprocessed into tensor format")
    parser.add_argument("--max-context-length", type=int, default=const.MAX_CONTEXT_LENGTH, help="Maximum length of a document")
    parser.add_argument("--max-query-length", type=int, default=const.MAX_QUERY_LENGTH, help="Maximum length of a question")
    parser.add_argument("--batch-size", type=int, default=const.BATCH_SIZE, help="Batch size on each compute device")
    parser.add_argument("--epochs", type=int, default=const.EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=const.LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=const.WARMUP_STEPS)
    parser.add_argument("--adam-eps", type=float, default=const.ADAM_EPS)
    parser.add_argument("--adam-betas", type=eval, default=const.ADAM_BETAS)
    parser.add_argument("--weight-decay", type=float, default=const.WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=const.MAX_GRAD_NORM)
    parser.add_argument("--shuffle", type=eval, default=const.SHUFFLE)
    parser.add_argument("--seed", type=int, default=const.SHUFFLE_SEED)
    parser.add_argument("--checkpoint-path", type=str, default=const.CHECKPOINT_PATH)
    parser.add_argument("--ctx-encoder-trainable", type=eval, default=const.CTX_ENCODER_TRAINABLE, help="Whether the context encoder's weights are trainable")
    parser.add_argument("--question-encoder-trainable", type=eval, default=const.QUESTION_ENCODER_TRAINABLE, help="Whether the question encoder's weights are trainable")
    parser.add_argument("--tpu", type=str, default="tpu-v3")
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")

    args = parser.parse_args()
    epochs = args.epochs

    try: # detect TPUs
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu) # TPU detection
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError: # detect GPUs
        devices = tf.config.list_physical_devices("GPU")
        # [tf.config.experimental.set_memory_growth(device, True) for device in devices]
        if devices:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

    tf.random.set_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    """Data pipeline
    1. Load retriever data (in `.tfrecord` format, stored serialized `tf.int32` tensor)
    2. Padding sequence to the same length
    3. Shuffle: You should shuffle before batch to guarantee that each data sample can be batched with different data samples in different epochs
    4. Repeating data: repeat to produce indefininte data stream
    5. Batching dataset
    6. Prefetching dataset (to speed up training) 
    """
    dataset = manipulator.load_retriever_tfrecord_int_data(
        input_path=args.data_path,
        shuffle=args.shuffle,
        shuffle_seed=args.seed
    )
    dataset = manipulator.pad(
        dataset, 
        sep_token_id=tokenizer.sep_token_id,
        max_context_length=args.max_context_length, 
        max_query_length=args.max_query_length
    )
    dataset = dataset.shuffle(buffer_size=60000)
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    """
    Distribute the dataset
    """
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )
    iterator = iter(dist_dataset)

    """
    Set up for distributed training
    """
    config = BertConfig.from_pretrained(
        args.pretrained_model,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )

    steps_per_epoch = args.train_data_size // (args.batch_size * strategy.num_replicas_in_sync)
    with strategy.scope():
        # Instantiate question encoder
        question_encoder = TFBertModel.from_pretrained(
            args.pretrained_model,
            config=config,
            trainable=args.question_encoder_trainable,
        )

        # Instantiate context encoder
        context_encoder = TFBertModel.from_pretrained(
            args.pretrained_model,
            config=config,
            trainable=args.ctx_encoder_trainable,
        )

        retriever = models.BiEncoder(
            question_model=question_encoder,
            ctx_model=context_encoder,
        )

        # Instantiate the optimizer
        optimizer = optimizers.get_adamw(
            steps_per_epoch=steps_per_epoch,
            warmup_steps=args.warmup_steps,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            eps=args.adam_eps,
            beta_1=args.adam_betas[0],
            beta_2=args.adam_betas[1],
            weight_decay=args.weight_decay,
        )

        # Define loss function
        loss_fn = losses.ThreeLevelDPRLoss(batch_size=args.batch_size)


    """
    Distributed train step
    """
    @tf.function
    def dist_train_step(element):
        """The step function for one training step"""
        print("This function is tracing !")

        def step_fn(element):
            """The computation to be run on each compute device"""
            question_ids = element['question']
            question_masks = tf.cast(question_ids > 0, tf.int32)
            contexts = element['contexts']
            context_ids = tf.reshape(contexts, [-1, tf.shape(contexts)[-1]])
            context_masks = tf.cast(context_ids > 0, tf.int32)

            with tf.GradientTape() as tape:
                q_tensors, ctx_tensors = retriever(
                    question_ids=question_ids,
                    question_masks=question_masks,
                    context_ids=context_ids,
                    context_masks=context_masks,
                    training=True
                )
                loss = loss_fn(q_tensors, ctx_tensors)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size * strategy.num_replicas_in_sync)

            grads = tape.gradient(loss, retriever.trainable_weights)
            grads = [tf.clip_by_norm(g, args.max_grad_norm) for g in grads]
            optimizer.apply_gradients(zip(grads, retriever.trainable_weights))

            return loss

        per_replica_losses = strategy.run(step_fn, args=(element,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    """
    Configure checkpoint
    """
    with strategy.scope():
        checkpoint_path = args.checkpoint_path
        ckpt = tf.train.Checkpoint(
            model=retriever,
            optimizer=optimizer,
            current_epoch=tf.Variable(0)
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

        # if a checkpoint exists, restore the latest checkpoint
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            current_epoch = ckpt.current_epoch.numpy()
            print("Latest checkpoint restored -- Model trained for {} epochs".format(current_epoch))
        else:
            print("Checkpoint not found. Train from scratch")
            current_epoch = 0


    """
    Bootstrap
    """
    sample = next(iter(dist_dataset))
    dist_train_step(sample)


    """
    Training loop
    """
    for epoch in range(current_epoch, epochs):
        print("*************** Epoch {:02d}/{:02d} ***************".format(epoch + 1, epochs))
        begin_epoch_time = time.perf_counter()

        for step in range(steps_per_epoch):
            begin_step_time = time.perf_counter()
            loss = dist_train_step(next(iterator))
            print("Step {: <6d}Loss: {: <20f}Elapsed: {}".format(
                step + 1,
                loss.numpy(),
                time.perf_counter() - begin_step_time,
            ))

        print("\nEpoch's elapsed time: {}\n".format(time.perf_counter() - begin_epoch_time))

        ckpt.current_epoch.assign_add(1)
        
        # Checkpoint the model
        ckpt_save_path = ckpt_manager.save()
        print ('\nSaving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))


if __name__ == "__main__":
    main()