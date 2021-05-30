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
