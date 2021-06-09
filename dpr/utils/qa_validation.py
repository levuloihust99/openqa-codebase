import unicodedata
import string
import regex as re
from typing import List, Dict, Text, Tuple
from multiprocessing import Pool as ProcessPool
from functools import partial
import collections
import numpy as np

from dpr.utils.tokenizers import SimpleTokenizer


QAMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_hits", "questions_doc_hits"]
)

def calculate_matches(
    all_docs: Dict[Text, Tuple[Text, Text]],
    closest_docs: List[Tuple[List[Text], np.ndarray]],
    answers: List[List[Text]],
    worker_num: int
):
    global dpr_all_documents
    dpr_all_documents = all_docs

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=worker_num)
    get_score_partial = partial(
        check_answer, tokenizer=tokenizer
    )

    closest_ids = [doc[0] for doc in closest_docs]
    answers_and_retrieved_docs = zip(answers, closest_ids)

    scores = processes.map(get_score_partial, answers_and_retrieved_docs)

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(
    answers_and_retrieved_docs: Tuple[List[Text], List[Text]],
    tokenizer
):
    """Search through the retrieved top-k documents to see if they have any of the answers."""
    global dpr_all_documents
    answers, closest_ids = answers_and_retrieved_docs

    hits = []
    for i, doc_id in enumerate(closest_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None: # cannot find the document for some reason
            print("No doc in database")
            hits.append(False)

        if has_answer(answers, text, tokenizer):
            answer_found = True
        hits.append(answer_found)

    return hits


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)

    text = tokenizer.tokenize(text).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i : i + len(single_answer)]:
                return True

    return False


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
