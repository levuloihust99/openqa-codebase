import tensorflow as tf
from transformers import TFBertModel, BertTokenizer


def base_compare():
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    sentence = "who got the first nobel prize in physics"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    token_ids = tokenizer.encode(sentence)
    token_ids = token_ids + [tokenizer.pad_token_id] * max(0, 256 - len(token_ids))

    input_ids = tf.convert_to_tensor([token_ids], dtype=tf.int32)
    attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)
    token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        training=False
    )

    pooled = outputs[0][:, 0, :]
    print("done")


def trained_compare():
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    retriever = tf.train.Checkpoint(ctx_model=encoder)
    ckpt = tf.train.Checkpoint(model=retriever)
    ckpt.restore("checkpoints/baseline/ckpt-1")

    sentence = "who got the first nobel prize in physics"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    token_ids = tokenizer.encode(sentence)
    token_ids = token_ids + [tokenizer.pad_token_id] * max(0, 256 - len(token_ids))
    token_ids[-1] = tokenizer.sep_token_id

    input_ids = tf.convert_to_tensor([token_ids], dtype=tf.int32)
    attention_mask = tf.cast(input_ids > 0, dtype=tf.int32)
    token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        training=False
    )

    pooled = outputs[0][:, 0, :]
    print("done")


if __name__ == "__main__":
    # base_compare()
    trained_compare()