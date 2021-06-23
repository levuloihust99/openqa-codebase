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
    --train-data-size 1000 \
    --data-path gs://openqa-dpr/data/retriever/vicovid/INT \
    --max-context-length 256 \
    --max-query-length 64 \
    --batch-size 8 \
    --epochs $run_epochs \
    --learning-rate 2e-5 \
    --warmup-steps 0 \
    --adam-eps 1e-8 \
    --adam-betas "(0.9, 0.999)" \
    --weight-decay 0.0 \
    --max-grad-norm 3.0 \
    --shuffle True \
    --seed 123 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/vicovid/vicovid_inbatch_batch8_query64_gradnorm3 \
    --ctx-encoder-trainable True \
    --question-encoder-trainable True \
    --tpu $run_tpu \
    --loss-fn inbatch \
    --use-pooler False \
    --load-optimizer True \
    --tokenizer NlpHUST/vibert4news-base-cased \
    --question-pretrained-model NlpHUST/vibert4news-base-cased \
    --context-pretrained-model NlpHUST/vibert4news-base-cased \
    --within-size 1 \
    --prefix pretrained