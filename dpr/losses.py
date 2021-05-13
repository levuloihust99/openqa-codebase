import tensorflow as tf

from typing import List
from . import const


class InBatchDPRLoss():
    def __init__(
        self,
        batch_size
    ):
        super(InBatchDPRLoss, self).__init__()
        self.batch_size = batch_size

    def __call__(
        self,
        q_tensors: tf.Tensor,
        ctx_tensors: tf.Tensor
    ):
        similarity_scores = tf.matmul(q_tensors, tf.transpose(ctx_tensors, perm=[1, 0]))
        log_softmax_scores = tf.math.log_softmax(similarity_scores, axis=-1)
        
        num_ctxs_per_question = 2
        indices = [[i, i * num_ctxs_per_question] for i in range(self.batch_size)]
        nll_loss = tf.gather_nd(log_softmax_scores, indices)

        return nll_loss * -1


class NLLDPRLoss():
    def __init__(self):
        super(NLLDPRLoss, self).__init__()
        self.binary_crossentropy_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def __call__(
        self,
        q_tensors: tf.Tensor,
        ctx_tensors: tf.Tensor,
        target_scores: tf.Tensor
    ):
        similarity_scores = tf.matmul(q_tensors, tf.transpose(ctx_tensors, perm=[1, 0]))
        normalized_scores = tf.nn.softmax(similarity_scores, axis=-1)
        return self.binary_crossentropy_loss_fn(target_scores, normalized_scores)