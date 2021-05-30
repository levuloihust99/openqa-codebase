#!/bin/bash
cd ~/openqa-codebase
python dense_retriever.py \
    --batch-size 1024 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/V3-base \
    --ctx-source-path data/wikipedia_split/psgs_subset.tsv \
    --disable-tf-function False \
    --embeddings-path data/retriever_results \
    --force-create-index False \
    --index-path indexer \
    --max-query-length 256 \
    --pretrained-model bert-base-uncased \
    --qas-tfrecord-path gs://openqa-dpr/data/qas/nq-test.tfrecord \
    --reader-data-path data/reader \
    --result-path results \
    --top-k 100 \
    --tpu tpu-v3