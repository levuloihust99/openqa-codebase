#!/bin/bash
cd ~/openqa-codebase
python validate_reader.py \
    --data-path gs://openqa-dpr/data/reader/nq/dev/dev.tfrecord \
    --max-sequence-length 256 \
    --tpu tpu-v3 \
    --pretrained-model bert-base-uncased \
    --batch-size 128 \
    --checkpoint-path gs://openqa-dpr/checkpoints/reader/baseline \
    --disable-tf-function False \
    --max-answer-length 10 \
    --res-dir results/reader/baseline