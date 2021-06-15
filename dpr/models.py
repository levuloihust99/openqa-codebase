from typing import Dict
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import TFBertModel, BertTokenizer, BertConfig

from typing import Dict, List, Tuple
from bigbird.core import modeling


class BiEncoder(keras.Model):
    def __init__(
        self,
        question_model: tf.keras.Model,
        ctx_model: tf.keras.Model,
        use_pooler: True,
        **kwargs
    ):
        super(BiEncoder, self).__init__(**kwargs)
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.use_pooler = use_pooler
    
    def call(
        self,
        question_inputs,
        context_inputs,
        **kwargs
    ):
        q_outputs = self.question_model(
            **question_inputs,
            **kwargs
        )
        q_sequence, q_pooled = q_outputs[0], q_outputs[1]

        ctx_outputs = self.ctx_model(
            **context_inputs,
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


def get_encoder(
    model_name,
    args,
    trainable: bool = True,
    prefix=None,
):
    MODEL_PATH = model_name
    if prefix is not None:
        MODEL_PATH = os.path.join(prefix, model_name)
    if model_name in {'bert-base-uncased', 'NlpHUST/vibert4news-base-cased'}:
        encoder = TFBertModel.from_pretrained(MODEL_PATH, trainable=trainable)
    elif model_name.find('bigbird') > -1:
        encoder = modeling.BertModel({
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 4096,
            "max_encoder_length": args.max_context_length,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "use_bias": True,
            "rescale_embedding": False,
            "use_gradient_checkpointing": False,
            "scope": "bert",
            "attention_type": "block_sparse",
            "norm_type": "postnorm",
            "block_size": 16,
            "num_rand_blocks": 3,
            "vocab_size": 50358
        })
        checkpoint_path = 'gs://bigbird-transformer/pretrain/bigbr_base/model.ckpt-0'
        checkpoint_reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
        encoder.set_weights([checkpoint_reader.get_tensor(v.name[:-2]) for v in encoder.trainable_weights])
        encoder.trainable = True
    else:
        raise Exception("Model {} not supported".format(model_name))

    if not args.use_pooler:
        if model_name in {'bert-base-uncased', 'NlpHUST/vibert4news-base-cased'}:
            encoder.bert.pooler.trainable = False
        elif model_name == 'bigbird':
            encoder.pooler.trainable = False

    return encoder


def get_tokenizer(
    model_name,
    prefix=None
):
    tokenizer_name = model_name
    if model_name == 'bigbird':
        tokenizer_name = 'bert-base-uncased'

    TOKENIZER_PATH = tokenizer_name
    if prefix is not None:
        TOKENIZER_PATH = os.path.join(prefix, model_name)

    return BertTokenizer.from_pretrained(TOKENIZER_PATH)


def get_config(
    model_name,
    prefix=None
):
    config_path = model_name
    if prefix is not None:
        config_path = os.path.join(prefix, model_name)
    if model_name in {'bert-base-uncased', 'NlpHUST/vibert4news-base-cased'}:
        return BertConfig.from_pretrained(config_path)
    else:
        raise Exception("Model {} is not supported".format(model_name))


def get_model_input(input_ids, attention_mask, model_name):
    if model_name in {'bert-base-uncased', 'NlpHUST/vibert4news-base-cased'}:
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    elif model_name == 'bigbird':
        return {
            'input_ids': input_ids
        }
    else:
        raise Exception("Model {} is not supported".format(model_name))