cd ~/openqa-codebase
if [ ! $epochs ]; then
    run_epochs=40
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

python train_retriever.py \
    --train-data-size 60000 \
    --data-path gs://openqa-dpr/data/retriever/V3/N5000-INT \
    --max-context-length 256 \
    --max-query-length 32 \
    --batch-size 16 \
    --epochs $run_epochs \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --adam-eps 1e-8 \
    --adam-betas "(0.9, 0.999)" \
    --weight-decay 0.0 \
    --max-grad-norm 2.0 \
    --shuffle True \
    --seed 123 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/threelevelsoftmax_batch16_query32 \
    --ctx-encoder-trainable True \
    --question-encoder-trainable True \
    --tpu $run_tpu \
    --pretrained-model bert-base-uncased \
    --loss-fn threelevelsoftmax \
    --use-pooler False \
    --load-optimizer True \
    --tokenizer bert-base-uncased \
    --within-size 8