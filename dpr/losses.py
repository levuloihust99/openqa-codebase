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


class ThreeLevelDPRLoss():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.binary_crossentropy_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        target_scores = tf.concat([tf.ones([7], dtype=tf.float32), tf.zeros([self.batch_size - 1], dtype=tf.float32)], axis=0)
        self.target_scores = tf.tile(tf.expand_dims(target_scores, axis=0), multiples=[self.batch_size, 1]) / 7
    
    def __call__(
        self,
        q_tensors: tf.Tensor,
        ctx_tensors: tf.Tensor
    ):
        # Positive vs hard negative (of the same sample)
        ctx_tensors_within = tf.reshape(ctx_tensors, [self.batch_size, 8, -1])
        q_tensors_within = tf.expand_dims(q_tensors, 1)
        scores_within = tf.matmul(q_tensors_within, tf.transpose(ctx_tensors_within, perm=[0, 2, 1]))
        scores_within = tf.squeeze(scores_within, axis=1)
        scores_within = -tf.math.log_softmax(scores_within, axis=-1)
        indices = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
        nll_loss_within = tf.gather_nd(scores_within, indices=indices)

        # Positive vs other positives (in batch)
        ctx_tensors_inbatch = ctx_tensors_within[:, 0, :]
        scores_inbatch = tf.matmul(q_tensors, tf.transpose(ctx_tensors_inbatch, perm=[1, 0]))
        scores_inbatch = -tf.math.log_softmax(scores_inbatch, axis=-1)
        indices = tf.tile(tf.expand_dims(tf.range(self.batch_size), axis=1), multiples=[1, 2])
        inbatch_loss = tf.gather_nd(scores_inbatch, indices=indices)

        # Hard negative vs other positives
        dim_0_indices_within = tf.tile(tf.expand_dims(tf.range(self.batch_size), axis=1), multiples=[1, 7])
        dim_1_indices_within = tf.tile(tf.expand_dims(tf.range(1, 8), axis=0), multiples=[self.batch_size, 1])
        indices_within = tf.concat([tf.expand_dims(dim_0_indices_within, axis=-1), tf.expand_dims(dim_1_indices_within, axis=-1)], axis=-1)
        scores_within_cut = tf.gather_nd(scores_within, indices_within)

        dim_0_indices_inbatch = tf.tile(tf.expand_dims(tf.range(self.batch_size), axis=1), multiples=[1, self.batch_size - 1])
        dim_1_indices_inbatch = [tf.concat([tf.range(i), tf.range(i + 1, self.batch_size)], axis=0) for i in range(self.batch_size)]
        dim_1_indices_inbatch = tf.convert_to_tensor(dim_1_indices_inbatch)
        indices_inbatch = tf.concat([tf.expand_dims(dim_0_indices_inbatch, axis=-1), tf.expand_dims(dim_1_indices_inbatch, axis=-1)], axis=-1)
        scores_inbatch_cut = tf.gather_nd(scores_inbatch, indices=indices_inbatch)
        
        scores_concat = tf.concat([scores_within_cut, scores_inbatch_cut], axis=-1)
        scores_concat = tf.math.softmax(scores_concat, axis=-1)

        binary_loss = self.binary_crossentropy_loss_fn(self.target_scores, scores_concat)
        return nll_loss_within + inbatch_loss + binary_loss


class TwoLevelDPRLoss():
    def __init__(self, batch_size, within_size):
        self.batch_size = batch_size
        self.within_size = within_size
        self.binary_crossentropy_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    
    def __call__(
        self,
        q_tensors: tf.Tensor,
        ctx_tensors: tf.Tensor
    ):
        # Positive vs hard negative (of the same sample)
        ctx_tensors_within = tf.reshape(ctx_tensors, [self.batch_size, self.within_size, -1])
        q_tensors_within = tf.expand_dims(q_tensors, 1)
        scores_within = tf.matmul(q_tensors_within, tf.transpose(ctx_tensors_within, perm=[0, 2, 1]))
        scores_within = tf.squeeze(scores_within, axis=1)
        scores_within = -tf.math.log_softmax(scores_within, axis=-1)
        indices = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
        nll_loss_within = tf.gather_nd(scores_within, indices=indices)

        # Positive vs other positives (in batch)
        ctx_tensors_inbatch = ctx_tensors_within[:, 0, :]
        scores_inbatch = tf.matmul(q_tensors, tf.transpose(ctx_tensors_inbatch, perm=[1, 0]))
        scores_inbatch = -tf.math.log_softmax(scores_inbatch, axis=-1)
        indices = tf.tile(tf.expand_dims(tf.range(self.batch_size), axis=1), multiples=[1, 2])
        inbatch_loss = tf.gather_nd(scores_inbatch, indices=indices)

        return nll_loss_within + inbatch_loss

