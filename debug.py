import tensorflow as tf
from transformers import TFBertModel
from dpr.models import Ranker, Reader
from dpr.losses.reader import ReaderLossCalculator


def reader():
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    batch_size = 4
    hidden_dim = 768
    max_sequence_length = 256
    max_answers = 10
    input_ids = tf.random.uniform([batch_size, max_sequence_length], maxval=1000, dtype=tf.int32)
    attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)
    reader = Reader(
        encoder=encoder,
        initializer_range=0.02,
    )
    start_logits, end_logits = reader.call(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    start_positions = tf.random.uniform([batch_size, max_answers], maxval=256, dtype=tf.int32)
    end_positions = tf.random.uniform([batch_size, max_answers], maxval=256, dtype=tf.int32)
    answer_mask = tf.cast(start_positions > 0, dtype=tf.float32)
    loss_calculator = ReaderLossCalculator()
    loss = loss_calculator.compute_token_loss(
        start_logits=start_logits,
        end_logits=end_logits,
        start_positions=start_positions,
        end_positions=end_positions,
        answer_mask=answer_mask
    )

    print("done")
    

def ranker():
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    batch_size = 2
    hidden_dim = 768
    psgs_per_question = 4
    input_ids = tf.random.uniform([batch_size, psgs_per_question, ])

if __name__ == "__main__":
    reader()