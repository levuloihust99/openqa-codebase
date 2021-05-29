#!/bin/bash
cd ~/openqa-codebase
python generate_embeddings.sh \
    --batch-size 32 \
    --checkpoint-path checkpoints/retriever/baseline \
    --ctx-source-shards-tfrecord data/wikipedia_split/shards-42031-tfrecord \
    --embeddings-path data/retriever_results \
    --max-context-length 256 \
    --pretrained-model bert-base-uncased \
    --records-per-file 42031 \
    --use-pooler False