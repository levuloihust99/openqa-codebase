cd ~/openqa-codebase
python train_retriever.py \
    --train-data-size 60000 \
    --data-path gs://openqa-dpr/data/retriever/V3/N5000-INT \
    --max-context-length 256 \
    --max-query-length 256 \
    --batch-size 16 \
    --epochs 40 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --adam-eps 1e-8 \
    --adam-betas "(0.9, 0.999)" \
    --weight-decay 0.0 \
    --max-grad-norm 2.0 \
    --shuffle True \
    --seed 123 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/default \
    --ctx-encoder-trainable True \
    --question-encoder-trainable True \
    --tpu tpu-v2-luffy \
    --pretrained-model bert-base-uncased \
    --loss-fn inbatch \
    --use-pooler False