from typing import List, Dict, Tuple, Text, Optional
from .qa_validation import normalize_answer


def get_best_span(
    start_logits,
    end_logits,
    sequence_ids,
    max_answers,
    tokenizer
):
    scores = []
    for (i, s) in enumerate(start_logits):
        for (j, e) in enumerate(end_logits[i : i + max_answers]):
            scores.append((i, i + j), s + e)

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    (start_index, end_index), _ = scores[0]

    # extend bpe subtokens to full tokens
    start_index, end_index = extend_span_to_full_words(
        tokenizer=tokenizer,
        sequence_ids=sequence_ids,
        span=(start_index, end_index)
    )

    predicted_answer = tokenizer.decode(sequence_ids[start_index : end_index + 1], skip_special_tokens=True)
    return predicted_answer


def compare_spans(
    answers: List[Text],
    span: Text
):
    return any([normalize_answer(span) == normalize_answer(answer) for answer in answers])


def extend_span_to_full_words(
    tokenizer,
    sequence_ids: List[int],
    span: Tuple[int, int],
    passage_offset: int
) -> Tuple[int, int]:
    start_index, end_index = span
    max_len = len(sequence_ids)
    while start_index > passage_offset and is_sub_word_id(tokenizer, sequence_ids[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and is_sub_word_id(tokenizer, sequence_ids[end_index + 1]):
        end_index += 1

    return start_index, end_index


def is_sub_word_id(
    tokenizer,
    token_id: int
):
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    return token.startswith("##") or token.startswith(" ##")