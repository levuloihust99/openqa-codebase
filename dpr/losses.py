import tensorflow as tf

from typing import List
from . import const

def dpr_loss(
    q_tensors: tf.Tensor,
    ctx_tensors: tf.Tensor,
):
    similarity_scores = tf.matmul(q_tensors, tf.transpose(ctx_tensors, perm=[1, 0]))
    log_softmax_scores = tf.math.log_softmax(similarity_scores, axis=-1)
    
    num_ctxs_per_question = 2
    indices = [[i, i * num_ctxs_per_question] for i in range(const.BATCH_SIZE)]
    nll_loss = tf.gather_nd(log_softmax_scores, indices)

    return nll_loss * -1