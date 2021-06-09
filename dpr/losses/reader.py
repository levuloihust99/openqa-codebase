import tensorflow as tf


class ReaderLossCalculator():
    def __init__(self):
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

    def compute_rank_loss(self, rank_logits, target):
        """
        Args:
            rank_logits: [batch_size, passages_per_sample]
            target: [batch_size] or [batch_size, passages_per_sample] (one hot)
        """
        # convert to onehot
        return self.cross_entropy(target, rank_logits)

    def compute_token_loss(
        self,
        start_logits,
        end_logits,
        start_positions,
        end_positions,
        answer_mask
    ):
        """
        Args
            start_logits: [batch_size, max_sequence_length]
            end_logits: [batch_sizee, max_sequence_length]
            start_positions: [batch_size, max_answers]
            end_positions: [batch_size, max_answers]
        """
        start_logits_extend = tf.expand_dims(start_logits, axis=1)
        start_logits_tile = tf.tile(start_logits_extend, multiples=[1, tf.shape(start_positions)[1], 1])
        end_logits_extend = tf.expand_dims(end_logits, axis=1)
        end_logits_tile = tf.tile(end_logits_extend, multiples=[1, tf.shape(end_positions)[1], 1])

        start_positions_onehot = tf.one_hot(start_positions, depth=tf.shape(start_logits)[1])
        end_positions_onehot = tf.one_hot(end_positions, depth=tf.shape(end_logits)[1])

        start_losses = self.cross_entropy(start_positions_onehot, start_logits_tile)
        end_losses = self.cross_entropy(end_positions_onehot, end_logits_tile)
        whole_losses = start_losses + end_losses
        whole_losses = whole_losses * answer_mask
        return tf.reduce_sum(whole_losses, axis=-1)