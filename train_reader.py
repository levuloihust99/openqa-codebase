import argparse
import os
from datetime import datetime
import copy
import time

import tensorflow as tf
from transformers import TFBertModel, BertConfig, BertTokenizer

from dpr.data import reader_manipulator
from utilities import write_config
from dpr import models, optimizers
from dpr.losses.reader import ReaderLossCalculator
from dpr.models import get_config, get_encoder, get_tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data-size", type=int, default=263011)
    parser.add_argument("--data-path", type=str, default="gs://openqa-dpr/data/reader/nq/train")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--max-answers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-betas", type=eval, default=(0.9, 0.999))
    parser.add_argument("--shuffle", type=eval, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint-path", type=str, default="gs://openqa-dpr/checkpoints/reader/baseline")
    parser.add_argument("--tpu", type=str, default="tpu-v3")
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")
    parser.add_argument("--load-optimizer", type=eval, default=True)
    parser.add_argument("--max-to-keep", type=int, default=50)
    parser.add_argument("--prefix", type=str, default='pretrained')

    args = parser.parse_args()
    args_dict = args.__dict__

    configs = ["{}: {}".format(k, v) for k, v in args_dict.items()]
    configs_string = "\t" + "\n\t".join(configs) + "\n"
    print("************************* Configurations *************************")
    print(configs_string)
    print("----------------------------------------------------------------------------------------------------------------------")

    config_path = "configs/{}/{}/config.yml".format(os.path.basename(__file__).rstrip(".py"), datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    write_config(config_path, args_dict)

    """
    Set up devices
    """
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

    tf.random.set_seed(args.seed)

    """
    Data pipeline
    """
    dataset = reader_manipulator.load_tfrecord_reader_train_data(
        input_path=args.data_path
    )

    dataset = reader_manipulator.transform_to_reader_train_dataset(
        dataset=dataset,
        max_sequence_length=256,
        max_answers=10
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=70000)
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    """
    Distribute the dataset
    """
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )
    iterator = iter(dataset)

    """
    Set up for distributed training
    """
    steps_per_epoch = args.train_data_size // (args.batch_size * strategy.num_replicas_in_sync)
    config = get_config(model_name=args.pretrained_model, prefix=args.prefix)
    with strategy.scope():
        encoder = get_encoder(
            model_name=args.pretrained_model,
            args=args,
            trainable=True,
            prefix=args.prefix
        )
        encoder.bert.pooler.trainable = False

        reader = models.Reader(
            encoder=encoder,
            initializer_range=config.initializer_range,
        )

        optimizer = optimizers.get_adamw(
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.adam_eps,
            beta_1=args.adam_betas[0],
            beta_2=args.adam_betas[1],
        )

        loss_calculator = ReaderLossCalculator()

    """
    Distributed train step
    """
    @tf.function
    def dist_train_step(element):
        """The step function for one training step"""
        print("This function is tracing !")

        def step_fn(element):
            """The computation to be run on each compute device"""
            input_ids = element['input_ids']
            attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)
            start_positions = element['start_positions']
            end_positions = element['end_positions']
            answer_mask = tf.cast(start_positions > 0, dtype=tf.float32)

            with tf.GradientTape() as tape:
                start_logits, end_logits = reader(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    training=True
                )

                loss = loss_calculator.compute_token_loss(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    answer_mask=answer_mask
                )

                loss = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size * strategy.num_replicas_in_sync)

            grads = tape.gradient(loss, reader.trainable_weights)
            grads = [tf.clip_by_norm(g, args.max_grad_norm) for g in grads]
            optimizer.apply_gradients(zip(grads, reader.trainable_weights))

            return loss

        per_replica_losses = strategy.run(step_fn, args=(element,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    """
    Configure checkpoint
    """
    with strategy.scope():
        checkpoint_path = args.checkpoint_path
        ckpt = tf.train.Checkpoint(
            model=reader,
            current_epoch=tf.Variable(0)
        )
        if not args.load_optimizer:
            tmp_optimizer = copy.deepcopy(optimizer)
            ckpt.optimizer = tmp_optimizer
        else:
            ckpt.optimizer = optimizer

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=args.max_to_keep)

        # if a checkpoint exists, restore the latest checkpoint
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            current_epoch = ckpt.current_epoch.numpy()
            print("Latest checkpoint restored -- Model trained for {} epochs".format(current_epoch))
        else:
            print("Checkpoint not found. Train from scratch")
            current_epoch = 0
        
        if not args.load_optimizer:
            ckpt.optimizer = optimizer

    """
    Bootstrap
    """
    sample = next(iter(dist_dataset))
    dist_train_step(sample)

    """
    Training loop
    """
    for epoch in range(current_epoch, args.epochs):
        print("*************** Epoch {:02d}/{:02d} ***************".format(epoch + 1, args.epochs))
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