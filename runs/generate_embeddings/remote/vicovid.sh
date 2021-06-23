#!/bin/bash
cd ~/openqa-codebase
python generate_embeddings.py \
    --batch-size 1024 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/vicovid/vicovid_inbatch_batch8_query64_gradnorm3_V3 \
    --ctx-source-shards-tfrecord gs://openqa-dpr/data/wikipedia_split/vicovid/V3/tfrecord \
    --embeddings-path data/retriever_results \
    --max-context-length 256 \
    --pretrained-model NlpHUST/vibert4news-base-cased \
    --records-per-file 42031 \
    --disable-tf-function False \
    --tpu tpu-v3-sanji \
    --use-pooler False \
    --prefix pretrained