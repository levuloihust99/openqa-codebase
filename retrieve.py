import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # This prevents tensorflow from logging WARNING messages

import pickle
import glob
import sys
from tqdm import tqdm
import argparse

import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertModel, BertConfig
import numpy as np

from dpr.indexer import DenseFlatIndexer


parser = argparse.ArgumentParser()
parser.add_argument("--create-index", type=eval, default=False, help="Whether to create index from embedded passages")
parser.add_argument("--query-path", type=str, default="data/qas/nq-test.csv", help="Path to the queries used to test retriever")
parser.add_argument("--ctx-source-path", type=str, default="data/wikipedia_split/psgs_w100.tsv", help="Path to the file containg all passages")
parser.add_argument("--checkpoint-path", type=str, default="checkpoints", help="Path to the checkpointed model")
parser.add_argument("--top-k", type=int, default=100, help="Number of documents that expects to be returned by retriever")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size when embedding questions")
parser.add_argument("--index-path", type=str, default="indexer", help="Path to indexed database")

args = parser.parse_args()

"""
Create index
"""
print("Creating index... ", end="")
sys.stdout.flush()

index_path = args.index_path
indexer = DenseFlatIndexer(buffer_size=50000)

if args.create_index:

    # Load embeddings
    ctx_embedding_path_pattern = "data/retriever_results/wikipedia_passages_{}.pkl"
    embedding_files = [ctx_embedding_path_pattern.format(i) for i in range(500)]

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
    indexer.deserialize(index_path)

print("Number of vectors: {}".format(indexer.index.ntotal))
print("done !")
print("-----------------------------------------------------------")


"""
Load QAS data
"""

print("Loading QAS data... ", end="")
sys.stdout.flush()

query_path = args.query_path
qas = pd.read_csv(query_path, sep="\t", header=None, names=['question', 'answers'])
assert qas.shape[0] == 3610

questions = qas.question.tolist()
answers   = qas.answers.tolist()

print("done !")
print("-----------------------------------------------------------")


"""
Load checkpoint
"""

print("Loading checkpoint... ")

checkpoint_path = args.checkpoint_path

config = BertConfig.from_pretrained(
    "bert-base-uncased",
    output_attentions=False,
    output_hidden_states=False,
    use_cache=False,
    return_dict=True,
)

question_encoder = TFBertModel.from_pretrained(
    "bert-base-uncased",
    config=config,
    trainable=False
)

retriever = tf.train.Checkpoint(question_model=question_encoder)
root_ckpt = tf.train.Checkpoint(model=retriever)

root_ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

print("done !")
print("-----------------------------------------------------------")


"""
Prepare input for the model
"""

print("Preparing dataset for inference... ", end="")
sys.stdout.flush()

text_dataset = tf.data.Dataset.from_tensor_slices(questions)
def transform_to_tensors(
    dataset: tf.data.Dataset,
    tokenizer: BertTokenizer,
    max_query_length: int = 32,
):

    def _generate():
        for element in dataset:
            query = element.numpy().decode('utf-8')
            tokens = tokenizer.encode(query)

            ids = tokenizer.convert_tokens_to_ids(tokens)
            ids = tf.constant(ids)
            ids = tf.pad(tokens, [[0, max_query_length]])[:max_query_length]
            mask = tf.cast(ids > 0, tf.int32)

            yield {
                'input_ids': ids,
                'attention_mask': mask,
            }

    return tf.data.Dataset.from_generator(
        _generate,
        output_signature={
            'input_ids': tf.TensorSpec([max_query_length], dtype=tf.int32),
            'attention_mask': tf.TensorSpec([max_query_length], dtype=tf.int32),
        }
    )

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = transform_to_tensors(text_dataset, tokenizer)

batch_size = args.batch_size
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("done !")
print("-----------------------------------------------------------")

"""
Generate question embeddings
"""
print("Generate question embeddings... ")

question_embeddings = []
count = 0
for question in tqdm(dataset):
    input_ids = question['input_ids']
    attention_mask = question['attention_mask']

    outputs = question_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

    pooled = outputs.pooler_output
    question_embeddings.extend(pooled.numpy())
    count += 1
    if count % 10 == 0:
        print("Embedded {} questions".format(count * batch_size))


print("Generate question done !")
print("-----------------------------------------------------------")


"""
Search nearest neighbors
"""
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


"""
Save search results to disk
"""
print("Save search results... ")

out_dir = "search_results"
out_path = os.path.join(out_dir, "top_ids_and_scores.pkl")

with open(out_path, "wb") as writer:
    pickle.dump(top_ids_and_scores, writer)

print("Saving done !")
print("-----------------------------------------------------------") 
