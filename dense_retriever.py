import os

import pickle
import glob
import sys
from tqdm import tqdm
import argparse
from datetime import datetime

import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertModel, BertConfig
import numpy as np

from typing import List, Any, Text, Tuple, Dict
from multiprocessing import Pool as ProcessPool
from functools import partial
import time
import json

from dpr.indexer import DenseFlatIndexer
from dpr import const
from dpr.utils.qa_validation import calculate_matches
from dpr.data import biencoder_manipulator
from utilities import write_config, spread_samples_greedy, spread_samples_equally
from dpr.models import get_encoder, get_tokenizer, get_model_input


def create_or_retrieve_indexer(index_path, embeddings_path):
    indexer = DenseFlatIndexer(buffer_size=50000)
    index_files = glob.glob("{}/*".format(index_path))

    if not index_files or args.force_create_index:
        if not index_files:
            print("Found no existing indexer. Creating new one...")
        else:
            print("Index exists, try to re-creating index...")

        indexer = DenseFlatIndexer(buffer_size=50000)

        print("Load and embed vectors... ")
        # Load embeddings
        # ctx_embedding_path_pattern = "{}/wikipedia_passages_{{}}.pkl".format(embeddings_path)
        # embedding_files = [ctx_embedding_path_pattern.format(i) for i in range(4)]
        ctx_embedding_path_pattern = "{}/wikipedia_passages_*.pkl".format(embeddings_path)
        embedding_files = glob.glob(ctx_embedding_path_pattern)
        embedding_files.sort()

        # Index data
        indexer.init_index(vector_sz=768)

        for embedding_file in tqdm(embedding_files):
            with open(embedding_file, "rb") as reader:
                vectors_shard = pickle.load(reader)
            vectors_shard = [(passage[0], passage[1].numpy()) for passage in vectors_shard]

            indexer.index_data(vectors_shard)

        # Write index to disk
        indexer.serialize(index_path)

    else:
        print("Indexer is already created...")
        print("Deserializing indexer from disk... ")
        indexer.deserialize(index_path)

    print("done")
    print("-----------------------------------------------------------")

    return indexer


def load_checkpoint(checkpoint_path, strategy):
    print("Loading checkpoint... ")

    with strategy.scope():
        question_encoder = get_encoder(
            model_name=args.pretrained_model,
            args=args,
            trainable=False,
            prefix=args.prefix
        )

        retriever = tf.train.Checkpoint(question_model=question_encoder)
        root_ckpt = tf.train.Checkpoint(model=retriever)

        root_ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

        print("Checkpoint file: {}".format(tf.train.latest_checkpoint(checkpoint_path)))
        print("done")
        print("-----------------------------------------------------------")

    return question_encoder


def load_qas_test_data():
    print("Loading QAS data... ", end="")
    sys.stdout.flush()

    query_path = args.query_path
    qas = pd.read_csv(query_path, sep="\t", header=None, names=['question', 'answers'])

    questions = qas.question.tolist()
    answers   = qas.answers.tolist()
    answers = [eval(answer) for answer in answers]

    print("done")
    print("-----------------------------------------------------------")

    return questions, answers


def prepare_dataset(qas_tfrecord_path, strategy, tokenizer, max_query_length: int = 256):
    print("Preparing dataset for inference... ", end="")
    sys.stdout.flush()

    dataset = biencoder_manipulator.load_tfrecord_tokenized_data_for_qas_ver2(
        qas_tfrecord_path=qas_tfrecord_path,
        sep_token_id=tokenizer.sep_token_id,
        max_query_length=max_query_length
    )
    dataset = dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )

    print("done !")
    print("-----------------------------------------------------------")
    return dataset


def generate_embeddings(
    question_encoder,
    dataset,
    strategy
):
    def dist_step(inputs):
        """The step function for one feed forward step"""
        if not args.disable_tf_function:
            print("This function is tracing !")

        def step_fn(inputs):
            """The computation to be run on each compute device"""
            outputs = question_encoder(
                **inputs,
                training=False
            )

            seq_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            if not args.use_pooler:
                pooled_output = seq_output[:, 0, :]
            return pooled_output
    
        per_replica_outputs = strategy.run(step_fn, args=(inputs,))
        return per_replica_outputs
    
    if not args.disable_tf_function:
        dist_step = tf.function(dist_step)

    def value_fn_template(ctx, indices, tensors):
        idxs = indices[ctx.replica_id_in_sync_group]
        return tensors[idxs[0] : idxs[1]]
        
    print("Generate question embeddings... ")

    count = 0
    question_embeddings = []
    for question in tqdm(dataset):
        if strategy.num_replicas_in_sync > 1:
            reduced_input_ids = tf.concat(question['question_ids'].values, axis=0)
        else:
            reduced_input_ids = question['question_ids']

        global_batch_size = reduced_input_ids.shape[0]
        if global_batch_size < args.batch_size * strategy.num_replicas_in_sync:
            if strategy.num_replicas_in_sync > 1:
                reduced_attention_mask = tf.concat(question['question_masks'].values, axis=0)
            else:
                reduced_attention_mask = question['question_masks']

            base_replica_batch_size = args.batch_size
            global_batch_outputs = None
            while True:
                spread, global_batch_size, base_replica_batch_size = spread_samples_equally(
                    global_batch_size=global_batch_size,
                    num_replicas=strategy.num_replicas_in_sync,
                    base_replica_batch_size=base_replica_batch_size,
                    init_batch_size=args.batch_size
                )

                if len(spread) > 1:
                    indices = []
                    idx = 0
                    for num in spread:
                        indices.append((idx, idx + num))
                        idx += num

                    value_fn_for_input_ids = partial(value_fn_template, indices=indices, tensors=reduced_input_ids)
                    value_fn_for_attention_mask = partial(value_fn_template, indices=indices, tensors=reduced_attention_mask)

                    reduced_input_ids = reduced_input_ids[base_replica_batch_size * strategy.num_replicas_in_sync:]
                    reduced_attention_mask = reduced_attention_mask[base_replica_batch_size * strategy.num_replicas_in_sync:]

                    dist_input_ids = strategy.experimental_distribute_values_from_function(value_fn_for_input_ids)
                    dist_attention_mask = strategy.experimental_distribute_values_from_function(value_fn_for_attention_mask)

                    question = get_model_input(
                        input_ids=dist_input_ids,
                        attention_mask=dist_attention_mask,
                        model_name=args.pretrained_model
                    )
                    per_replica_outputs = dist_step(question)
                    if global_batch_outputs is None:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = tf.concat(per_replica_outputs.values, axis=0)
                        else:
                            global_batch_outputs = per_replica_outputs
                    else:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = tf.concat([global_batch_outputs, *per_replica_outputs.values], axis=0)
                        else:
                            global_batch_outputs = tf.concat([global_batch_outputs, per_replica_outputs], axis=0)
                    
                    if global_batch_size == 0:
                        break

                else:
                    question = get_model_input(
                        input_ids=reduced_input_ids,
                        attention_mask=reduced_attention_mask,
                        model_name=args.pretrained_model
                    )

                    per_replica_outputs = dist_step(question)

                    if global_batch_outputs is None:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = per_replica_outputs.values[0]
                        else:
                            global_batch_outputs = per_replica_outputs
                    else:
                        if strategy.num_replicas_in_sync > 1:
                            global_batch_outputs = tf.concat([global_batch_outputs, per_replica_outputs.values[0]], axis=0)
                        else:
                            global_batch_outputs = tf.concat([global_batch_outputs, per_replica_outputs], axis=0)

                    break

            
        else:
            per_replica_outputs = dist_step(question)
            if strategy.num_replicas_in_sync > 1:
                global_batch_outputs = tf.concat(per_replica_outputs.values, axis=0)
            else:
                global_batch_outputs = per_replica_outputs

        question_embeddings.extend(global_batch_outputs.numpy())
        count += 1
        if count % 10 == 0:
            print("Embedded {} questions".format(len(question_embeddings)))

    print("Generate question done !")
    print("-----------------------------------------------------------")

    return question_embeddings


def search_knn(
    indexer,
    question_embeddings
):
    print("Searching top-k documents... ")

    top_docs = args.top_k
    query_vectors = np.array(question_embeddings)

    total_queries = query_vectors.shape[0]
    batch = total_queries // 20

    print("Batch search size: {}".format(batch))

    top_ids_and_scores = []
    for i in tqdm(range(0, total_queries, batch)):
        top_ids_and_scores_batch = indexer.search_knn(query_vectors[i: i + batch], top_docs)
        top_ids_and_scores.extend(top_ids_and_scores_batch)

    print("Search done !")
    print("-----------------------------------------------------------") 
    
    return top_ids_and_scores


def load_ctx_sources():
    print("Loading all documents into memory... ")
    start_time = time.perf_counter()
    all_docs_df = pd.read_csv(args.ctx_source_path, sep='\t', header=0)
    all_ids = all_docs_df.id.tolist()
    all_ids = ["wiki:{}".format(id) for id in all_ids]
    all_docs = dict(zip(all_ids, zip(all_docs_df.text, all_docs_df.title)))
    print("done in {}s".format(time.perf_counter() - start_time))
    print("----------------------------------------------------------------")

    return all_docs


def validate(
    top_ids_and_scores: List[Tuple[List, np.ndarray]],
    answers: List[List[Text]],
    ctx_sources: Dict[Text, Text],
    top_k_hits_path: Text
):
    match_stats = calculate_matches(
        all_docs=ctx_sources,
        closest_docs=top_ids_and_scores,
        answers=answers,
        worker_num=os.cpu_count()
    )

    top_k_hits = match_stats.top_k_hits
    n_queries = len(answers)
    top_k_hits = [v / n_queries for v in top_k_hits]

    stats = []
    for i, v in enumerate(top_k_hits):
        stats.append("Top {: <3} hits: {:.2f}".format(i + 1, v * 100))

    with open(top_k_hits_path, "w") as writer:
        writer.write("\n".join(stats))

    return match_stats.questions_doc_hits


def save_results(
    questions: List[Text],
    answers: List[List[Text]],
    all_docs: Dict[Text, Tuple[Text, Text]],
    top_passages_and_scores: List[Tuple[List[Text], np.ndarray]],
    per_question_hits: List[List[bool]],
    out_file: str
):

    reader_data = []
    for i, question in enumerate(questions):
        answer_list = answers[i]
        hits = per_question_hits[i]
        top_passages, top_scores = top_passages_and_scores[i]
        docs = [all_docs[doc] for doc in top_passages]

        reader_sample = {
            'question': question,
            'answers': answer_list,
            'ctxs': [
                {
                    'id': top_passages[c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': str(top_scores[c]),
                    'has_answer': hits[c]
                }
                for c in range(len(hits))
            ]
        }
        reader_data.append(reader_sample)

    with open(out_file, "w") as writer:
        json.dump(reader_data, writer, indent=4)
    print("Save reader data to {}".format(out_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-path", type=str, default=const.QUERY_PATH, help="Path to the queries used to test retriever")
    parser.add_argument("--ctx-source-path", type=str, default=const.CTX_SOURCE_PATH, help="Path to the file containg all passages")
    parser.add_argument("--checkpoint-path", type=str, default=const.CHECKPOINT_PATH, help="Path to the checkpointed model")
    parser.add_argument("--top-k", type=int, default=const.TOP_K, help="Number of documents that expects to be returned by retriever")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size when embedding questions")
    parser.add_argument("--index-path", type=str, default=const.INDEX_PATH, help="Path to indexed database")
    parser.add_argument("--pretrained-model", type=str, default=const.PRETRAINED_MODEL)
    parser.add_argument("--reader-data-path", type=str, default=const.READER_DATA_PATH)
    parser.add_argument("--result-path", type=str, default=const.RESULT_PATH)
    parser.add_argument("--embeddings-path", type=str, default=const.EMBEDDINGS_DIR)
    parser.add_argument("--force-create-index", type=eval, default=False)
    parser.add_argument("--qas-tfrecord-path", type=str, default=const.QAS_TFRECORD_PATH)
    parser.add_argument("--max-query-length", type=int, default=const.MAX_QUERY_LENGTH)
    parser.add_argument("--disable-tf-function", type=eval, default=False)
    parser.add_argument("--tpu", type=str, default=const.TPU_NAME)
    parser.add_argument("--use-pooler", type=eval, default=True)
    parser.add_argument("--prefix", type=str, default='pretrained')

    global args
    args = parser.parse_args()
    model_type = os.path.basename(args.checkpoint_path)
    embeddings_path = os.path.join(args.embeddings_path, "shards-42031", model_type)
    args_dict = {**args.__dict__, "embeddings_path" : embeddings_path}

    configs = ["{}: {}".format(k, v) for k, v in args_dict.items()]
    configs_string = "\t" + "\n\t".join(configs) + "\n"
    print("************************* Configurations *************************")
    print(configs_string)
    print("----------------------------------------------------------------------------------------------------------------------")

    config_path = "configs/{}/{}/config.yml".format(__file__.rstrip(".py"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    write_config(config_path, args_dict)

    try: # detect TPUs
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu) # TPU detection
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except Exception: # detect GPUs
        devices = tf.config.list_physical_devices("GPU")
        # [tf.config.experimental.set_memory_growth(device, True) for device in devices]
        if devices:
            strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        else:
            strategy = tf.distribute.get_strategy()

    index_path = os.path.join(args.index_path, model_type)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    indexer = create_or_retrieve_indexer(index_path=index_path, embeddings_path=embeddings_path)
    # exit(0) # only create index
    question_encoder = load_checkpoint(checkpoint_path=args.checkpoint_path, strategy=strategy)
    questions, answers = load_qas_test_data()
    tokenizer = get_tokenizer(model_name=args.pretrained_model, prefix=args.prefix)
    dataset = prepare_dataset(
        args.qas_tfrecord_path,
        strategy=strategy,
        tokenizer=tokenizer,
        max_query_length=args.max_query_length
    )
    question_embeddings = generate_embeddings(question_encoder=question_encoder, dataset=dataset, strategy=strategy)
    top_ids_and_scores = search_knn(indexer=indexer, question_embeddings=question_embeddings)
    all_docs = load_ctx_sources()

    print("Validating... ")
    top_k_hits_path = os.path.join(args.result_path, model_type, "top_k_hits.txt")
    if not os.path.exists(os.path.dirname(top_k_hits_path)):
        os.makedirs(os.path.dirname(top_k_hits_path))

    start_time = time.perf_counter()
    questions_doc_hits = validate(
        top_ids_and_scores=top_ids_and_scores,
        answers=answers,
        ctx_sources=all_docs,
        top_k_hits_path=top_k_hits_path
    )
    print("done in {}s !".format(time.perf_counter() - start_time))
    print("----------------------------------------------------------------------------------------------------------------------")

    print("Generating reader data... ")
    reader_data_path = os.path.join(args.reader_data_path, model_type, "reader_data.json")
    if not os.path.exists(os.path.dirname(reader_data_path)):
        os.makedirs(os.path.dirname(reader_data_path))
    start_time = time.perf_counter()
    save_results(
        questions=questions,
        answers=answers,
        all_docs=all_docs,
        top_passages_and_scores=top_ids_and_scores,
        per_question_hits=questions_doc_hits,
        out_file=reader_data_path
    )

    print("done in {}s !".format(time.perf_counter() - start_time))
    print("----------------------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()