#!/bin/bash
cd ~/openqa-codebase
python dense_retriever.py \
    --batch-size 1024 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/vicovid/vicovid_inbatch_batch8_query64_gradnorm3 \
    --ctx-source-path data/wikipedia_split/V2/vicovid_ctx_sources.tsv \
    --disable-tf-function False \
    --embeddings-path data/retriever_results \
    --force-create-index True \
    --index-path indexer \
    --max-query-length 64 \
    --pretrained-model NlpHUST/vibert4news-base-cased \
    --qas-tfrecord-path gs://openqa-dpr/data/qas/nq-test-ver2.tfrecord \
    --reader-data-path data/reader \
    --result-path results \
    --top-k 100 \
    --tpu tpu-v3-nami \
    --use-pooler False