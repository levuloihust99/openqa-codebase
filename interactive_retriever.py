from numpy.core.fromnumeric import argsort
from dpr.indexer import DenseFlatIndexer
import glob
from argparse import Namespace
from dpr.models import get_encoder, get_tokenizer
import tensorflow as tf
import time
import pandas as pd
from functools import partial


def load_checkpoint(
    args,
    checkpoint_path: str = "gs://openqa-dpr/retriever/checkpoints/vi-covid/vicovid_inbatch_batch8_query64",
):
    question_encoder = get_encoder(
        model_name=args.pretrained_model,
        args=args,
        trainable=False,
        prefix='pretrained'
    )
    retriever = tf.train.Checkpoint(question_model=question_encoder)
    root_ckpt = tf.train.Checkpoint(model=retriever)

    root_ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    print("Checkpoint: {}".format(tf.train.latest_checkpoint(checkpoint_path)))
    return question_encoder


def load_ctx_sources(ctx_source_path):
    print("Loading all documents into memory... ")
    start_time = time.perf_counter()
    all_docs_df = pd.read_csv(ctx_source_path, sep='\t', header=0)
    all_ids = all_docs_df.id.tolist()
    all_ids = ["wiki:{}".format(id) for id in all_ids]
    all_docs = dict(zip(all_ids, zip(all_docs_df.text, all_docs_df.title)))
    print("done in {}s".format(time.perf_counter() - start_time))
    print("----------------------------------------------------------------")

    return all_docs


def get_query_vectors(
    question_encoder,
    input_question,
    tokenizer
):
    tokens = tokenizer.tokenize(input_question)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
    token_ids = token_ids[:args.max_query_length]
    token_ids = token_ids + [tokenizer.pad_token_id] * (args.max_query_length - len(token_ids))

    input_ids = tf.convert_to_tensor([token_ids], dtype=tf.int32)
    attention_mask = tf.cast(input_ids != tokenizer.pad_token_id, dtype=tf.int32)
    outputs = question_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    pooled_output = outputs[0][:, 0, :]
    return pooled_output.numpy()


def retrieve(input_question, indexer, question_encoder, tokenizer, top_docs, all_docs):
    query_vectors = get_query_vectors(question_encoder, input_question, tokenizer)
    top_ids_and_scores = indexer.search_knn(query_vectors, top_docs=top_docs)
    top_ids = top_ids_and_scores[0][0]
    retrieved_docs = [all_docs[id] for id in top_ids]
    for doc in retrieved_docs:
        print(" || ".join([doc[0], doc[1]]))
        print("--------------------------------------------------------------------------------")


if __name__ == "__main__":
    global args
    kwargs = {
        'ctx_source_path': 'data/wikipedia_split/vi_covid_subset_ctx_source.tsv',
        'pretrained_model': 'NlpHUST/vibert4news-base-cased',
        'use_pooler': False,
        'max_query_length': 64
    }
    args = Namespace(**kwargs)

    index_path = "indexer/vicovid_inbatch_batch8_query64"
    indexer = DenseFlatIndexer(buffer_size=50000)
    index_files = glob.glob("{}/*".format(index_path))
    print("Deserializing indexer from disk... ")
    indexer.deserialize(index_path)

    all_docs = load_ctx_sources(args.ctx_source_path)
    
    tokenizer = get_tokenizer(model_name='NlpHUST/vibert4news-base-cased', prefix='pretrained')
    question_encoder = load_checkpoint(args)
    
    tokenizer = get_tokenizer(model_name='NlpHUST/vibert4news-base-cased')

    retrieve_func = partial(retrieve, indexer=indexer, question_encoder=question_encoder, tokenizer=tokenizer, top_docs=10, all_docs=all_docs)
    
    print("done")