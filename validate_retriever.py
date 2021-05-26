import pickle
import argparse
import pandas as pd
import numpy as np
from typing import List, Any, Text, Tuple, Dict
from multiprocessing import Pool as ProcessPool
from functools import partial
import time
import json

from dpr.qa_validation import calculate_matches

parser = argparse.ArgumentParser()
parser.add_argument("--search-results", type=str, default="results/cache/search_results/top_ids_and_scores.pkl", help="Path to the top ids and scores returned by retriever")
parser.add_argument("--ctx-source-path", type=str, default="data/wikipedia_split/psgs_subset.tsv", help="Path to the file containing all passages")
parser.add_argument("--qas-path", type=str, default="data/qas/nq-test-subset.csv", help="Path to the queries used to test the retriever")
parser.add_argument("--reader-data-path", type=str, default="data/reader/reader_data_subset.json", help="File to write out reader data")

args = parser.parse_args()

with open(args.search_results, "rb") as reader:
    top_ids_and_scores = pickle.load(reader)

qas = pd.read_csv(args.qas_path, sep='\t', header=None, names=['question', 'answers'])
questions = qas.question.tolist()
answers = qas.answers.tolist()
answers = [eval(answer) for answer in answers]

print("Loading all documents into memory... ")
start_time = time.perf_counter()
all_docs_df = pd.read_csv(args.ctx_source_path, sep='\t', header=0)
all_ids = all_docs_df.id.tolist()
all_ids = ["wiki:{}".format(id) for id in all_ids]
all_docs = dict(zip(all_ids, zip(all_docs_df.text, all_docs_df.title)))
print("done in {}s !".format(time.perf_counter() - start_time))
print("----------------------------------------------------------------")

print("Validating... ")
def validate(
    top_ids_and_scores: List[Tuple[List, np.ndarray]],
    answers: List[List[Text]],
    ctx_sources: Dict[Text, Text]
):
    match_stats = calculate_matches(
        all_docs=ctx_sources,
        closest_docs=top_ids_and_scores,
        answers=answers,
        worker_num=24
    )

    top_k_hits = match_stats.top_k_hits
    n_queries = len(answers)
    top_k_hits = [v / n_queries for v in top_k_hits]

    stats = []
    for i, v in enumerate(top_k_hits):
        stats.append("Top {: <3} hits: {:.2f}".format(i + 1, v * 100))

    with open("results/top_k_hits.txt", "w") as writer:
        writer.write("\n".join(stats))

    return match_stats.questions_doc_hits

start_time = time.perf_counter()
questions_doc_hits = validate(top_ids_and_scores, answers, all_docs)
print("done in {}s !".format(time.perf_counter() - start_time))
print("----------------------------------------------------------------")

print("Generating reader data... ")
start_time = time.perf_counter()

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

save_results(
    questions=questions,
    answers=answers,
    all_docs=all_docs,
    top_passages_and_scores=top_ids_and_scores,
    per_question_hits=questions_doc_hits,
    out_file=args.reader_data_path
)

print("done in {}s !".format(time.perf_counter() - start_time))
print("----------------------------------------------------------------")
