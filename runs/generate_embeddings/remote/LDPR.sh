#!/bin/bash
cd ~/openqa-codebase
python generate_embeddings.py \
    --batch-size 1024 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/inbatch_batch16_query256_bigbird \
    --ctx-source-shards-tfrecord gs://openqa-dpr/data/wikipedia_split/shards-42031-tfrecord \
    --embeddings-path data/retriever_results \
    --max-context-length 256 \
    --pretrained-model bigbird \
    --records-per-file 42031 \
    --disable-tf-function False \
    --tpu tpu-v3-nami \
    --use-pooler False