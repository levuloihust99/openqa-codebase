#!/bin/bash
cd ~/openqa-codebase
python generate_embeddings.sh \
    --batch-size 1024 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/baseline \
    --ctx-source-shards-tfrecord gs://openqa-dpr/data/wikipedia_split/shards-42031-tfrecord \
    --embeddings-path data/retriever_results \
    --max-context-length 256 \
    --pretrained-model bert-base-uncased \
    --records-per-file 42031 \
    --tpu tpu-v3 \
    --use-pooler True