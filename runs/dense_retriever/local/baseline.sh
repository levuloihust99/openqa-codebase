#!/bin/bash
cd ~/openqa-codebase
python dense_retriever.py \
    --batch-size 16 \
    --checkpoint-path checkpoints/retriever/baseline \
    --ctx-source-path data/wikipedia_split/psgs_subset.tsv \
    --disable-tf-function True \
    --embeddings-path data/retriever_results \
    --force-create-index False \
    --index-path indexer \
    --max-query-length 256 \
    --pretrained-model bert-base-uncased \
    --qas-tfrecord-path data/qas/nq-test.tfrecord \
    --reader-data-path data/reader \
    --result-path results \
    --top-k 100 \
    --tpu tpu-v3 \
    --disable-tf-function False \
    --use-pooler False \