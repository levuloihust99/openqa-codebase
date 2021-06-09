#!/bin/bash
cd ~/openqa-codebase
python validate_endtoend.py \
    --data-path gs://openqa-dpr/data/reader/nq/dev/dev.tfrecord \
    --max-sequence-length 256 \
    --max-passages 50 \
    --max-answer-length 10 \
    --batch-size 4 \
    --tpu tpu-v3-sanji \
    --ranker-checkpoint-path gs://openqa-dpr/checkpoints/ranker/baseline \
    --reader-checkpoint-path gs://openqa-dpr/checkpoints/reader/baseline \
    --pretrained-model bert-base-uncased \
    --use-pooler False \
    --disable-tf-function False \
    --res-dir results/endtoend