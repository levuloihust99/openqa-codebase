from typing import Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import TFBertModel

from typing import Dict, List, Tuple


class BiEncoder(keras.Model):
    def __init__(
        self,
        question_model: TFBertModel,
        ctx_model: TFBertModel,
        use_pooler: True,
        **kwargs
    ):
        super(BiEncoder, self).__init__(**kwargs)
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.use_pooler = use_pooler
    
    def call(
        self,
        question_ids,
        question_masks,
        context_ids,
        context_masks,
        **kwargs
    ):
        q_outputs = self.question_model(
            input_ids=question_ids, 
            attention_mask=question_masks,
            **kwargs
        )
        q_sequence, q_pooled = q_outputs[0], q_outputs[1]

        ctx_outputs = self.ctx_model(
            input_ids=context_ids, 
            attention_mask=context_masks,
            **kwargs
        )
        ctx_sequence, ctx_pooled = ctx_outputs[0], ctx_outputs[1]
        if not self.use_pooler:
            q_pooled = q_sequence[:, 0, :]
            ctx_pooled = ctx_sequence[:, 0, :]

        return q_pooled, ctx_pooled


class Ranker(tf.keras.Model):
    def __init__(
        self,
        encoder,
        initializer_range: float,
        use_pooler: bool = False,
        **kwargs
    ):
        super(Ranker, self).__init__(**kwargs)
        self.encoder = encoder
        self.selector = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
        )
        self.use_pooler = use_pooler

    def call(
        self,
        input_ids,
        attention_mask,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output, pooled_output = outputs[0], outputs[1]
        if not self.use_pooler:
            pooled_output = sequence_output[:, 0, :]

        rank_logits = self.selector(pooled_output)
        rank_logits = tf.squeeze(rank_logits, axis=1)
        return rank_logits


class Reader(tf.keras.Model):
    def __init__(
        self,
        encoder,
        initializer_range: float,
        **kwargs
    ):
        super(Reader, self).__init__(**kwargs)
        self.encoder = encoder
        self.token_classifier = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
        )

    def call(
        self,
        input_ids,
        attention_mask,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        logits = self.token_classifier(sequence_output)
        start_logits, end_logits = tf.split(logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return start_logits, end_logits
