import numpy as np
import pickle
import collections

import torch
from torch.serialization import default_restore_location
import tensorflow as tf

from transformers import BertModel, TFBertModel, BertTokenizer

from dpr.models import BiEncoder


CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)


def check_compability(
    torch_model: BertModel,
    tf_model: TFBertModel
):
    torch_weights = []
    for k, v in torch_model.state_dict().items():
        if k == "embeddings.position_ids":
            print("im here")
            continue
        if not k.startswith("embeddings.") and k.endswith(".weight"):
            torch_weights.append(v.t().numpy())
        else:
            torch_weights.append(v.numpy())
    torch_weights[1], torch_weights[2] = torch_weights[2], torch_weights[1]

    tf_weights = tf_model.get_weights()

    check = [(torch_weight == tf_weight).all() for torch_weight, tf_weight in zip(torch_weights, tf_weights)]
    return all(check)


def direct_convert():
    state_dict = torch.load("checkpoints/nq/single/bert-base-encoder.cp", map_location=lambda s, l: default_restore_location(s, "cpu"))
    state_dict = CheckpointState(**state_dict)

    question_prefix = "question_model."
    ctx_prefix = "ctx_model."

    question_encoder_state = {
        key[len(question_prefix):]: value
        for key, value in state_dict.model_dict.items()
        if key.startswith(question_prefix)
    }
    torch_question_encoder = BertModel.from_pretrained('pretrained/bert-base-uncased')
    torch_question_encoder.load_state_dict(question_encoder_state, strict=False)

    ctx_encoder_state = {
        key[len(ctx_prefix):]: value
        for key, value in state_dict.model_dict.items()
        if key.startswith(ctx_prefix)
    }
    torch_ctx_encoder = BertModel.from_pretrained('pretrained/bert-base-uncased')
    torch_ctx_encoder.load_state_dict(ctx_encoder_state, strict=False)
    
    question_encoder_weights = []
    for k, v in question_encoder_state.items():
        if k.endswith(".weight") and not k.startswith("embeddings."):
            v = v.t()
        question_encoder_weights.append(v.numpy())
    question_encoder_weights[1], question_encoder_weights[2] = question_encoder_weights[2], question_encoder_weights[1]

    ctx_encoder_weights = []
    for k, v in ctx_encoder_state.items():
        if k.endswith(".weight") and not k.startswith("embeddings."):
            v = v.t()
        ctx_encoder_weights.append(v.numpy())
    ctx_encoder_weights[1], ctx_encoder_weights[2] = ctx_encoder_weights[2], ctx_encoder_weights[1]

    tf_question_encoder = TFBertModel.from_pretrained('pretrained/bert-base-uncased')
    tf_ctx_encoder = TFBertModel.from_pretrained('pretrained/bert-base-uncased')
    tf_question_encoder.set_weights(question_encoder_weights)
    tf_ctx_encoder.set_weights(ctx_encoder_weights)

    assert check_compability(torch_question_encoder, tf_question_encoder)
    assert check_compability(torch_ctx_encoder, tf_ctx_encoder)

    tf_biencoder = BiEncoder(question_model=tf_question_encoder, ctx_model=tf_ctx_encoder)
    tf_ckpt = tf.train.Checkpoint(model=tf_biencoder)
    tf_manager = tf.train.CheckpointManager(tf_ckpt, "checkpoints/nq/single/", max_to_keep=3)
    tf_manager.save()


if __name__ == "__main__":
    direct_convert()
