cd ~/openqa-codebase
if [ ! $epochs ]; then
    run_epochs=100
else
    run_epochs=$epochs
    unset epochs
fi

if [ ! $tpu ]; then
    run_tpu="tpu-v3"
else
    run_tpu=$tpu
    unset tpu
fi

python train_reader.py \
    --train-data-size 265000 \
    --data-path gs://openqa-dpr/data/reader/nq/train \
    --max-sequence-length 256 \
    --max-answers 10 \
    --batch-size 128 \
    --epochs $run_epochs \
    --learning-rate 2e-5 \
    --warmup-steps 0 \
    --adam-eps 1e-8 \
    --adam-betas "(0.9, 0.999)" \
    --weight-decay 0.0 \
    --max-grad-norm 2.0 \
    --shuffle True \
    --seed 123 \
    --checkpoint-path gs://openqa-dpr/checkpoints/reader/baseline \
    --tpu $run_tpu \
    --pretrained-model bert-base-uncased \
    --load-optimizer True \