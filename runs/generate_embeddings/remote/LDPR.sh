#!/bin/bash
cd ~/openqa-codebase
python generate_embeddings.py \
    --batch-size 1024 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/nq/baseline/single \
    --ctx-source-shards-tfrecord gs://openqa-dpr/data/wikipedia_split/nq/shards-42031-tfrecord \
    --embeddings-path data/retriever_results \
    --max-context-length 256 \
    --pretrained-model bert-base-uncased \
    --records-per-file 42031 \
    --disable-tf-function False \
    --tpu tpu-v3-luffy \
    --use-pooler False