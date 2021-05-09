import tensorflow as tf

from official import nlp
import official.nlp.optimization

import math
from typing import Dict


def get_adamw(
    epochs: int = 40,
    steps_per_epoch: int = 918,
    warmup_steps: int = 100,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    beta_1: float = 0.9,
    beta_2: float = 0.999
):
    num_train_steps = epochs * steps_per_epoch

    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=0
    )

    warmup_schedule = nlp.optimization.WarmUp(
        initial_learning_rate=learning_rate,
        decay_schedule_fn = decay_schedule,
        warmup_steps=warmup_steps
    )

    return nlp.optimization.AdamWeightDecay(
        learning_rate=warmup_schedule,
        weight_decay_rate=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=eps,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias']
    )
