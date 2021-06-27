# Requirements
This code base is tested on:
- Python 3.8.5
- Tensorflow 2.4.1
- Transformers 4.4.2

You also need to install Google Cloud SDK to read and write data stored on Google Cloud Storage

# Environment setup
Make sure `python3-dev`, `python3-venv` and `build-essential` are installed on your machine. On Ubuntu, run the following commands
```shell
$ sudo apt update
$ sudo apt install python3-dev python3-venv build-essential
```

Next, create a virtual environment and activate it
```shell
$ python3.8 -m venv .venv
$ source .venv/bin/activate
```

Install requirements for this codebase
```shell
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements.txt 
```

Download pretrained model
```shell
$ gsutil cp -r gs://openqa-dpr/pretrained/ .
```

# Training
This code can run on Cloud TPUs, multi-GPUs computer or a computer with CPU only.

Refer to files in `runs/` to see how to config parameters for training.

Training retriever example
```shell
$ tpu=tpu-v3 epochs=40 runs/train_retriever/hardnegvsnegsoftmax_batch16_query32_gradnorm3.sh
```

Running the above bash script is equivalent with running the following command

    python train_retriever.py \
    --train-data-size 60000 \
    --data-path gs://openqa-dpr/data/retriever/V3/N5000-INT \
    --max-context-length 256 \
    --max-query-length 32 \
    --batch-size 16 \
    --epochs 40 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --adam-eps 1e-8 \
    --adam-betas "(0.9, 0.999)" \
    --weight-decay 0.0 \
    --max-grad-norm 3.0 \
    --shuffle True \
    --seed 123 \
    --checkpoint-path gs://openqa-dpr/checkpoints/retriever/hardnegvsnegsoftmax_batch16_query32_gradnorm3 \
    --ctx-encoder-trainable True \
    --question-encoder-trainable True \
    --tpu tpu-v3 \
    --tokenizer bert-base-uncased \
    --question-pretrained-model bert-base-uncased \
    --context-pretrained-model bert-base-uncased \
    --prefix pretrained \
    --loss-fn hardnegvsnegsoftmax \
    --use-pooler False \
    --tokenizer bert-base-uncased \
    --load-optimizer True \
    --within-size 8

Training reader and re-ranker are also simply done by running the provided scripts. Have a look at `runs/generate_embeddings/`, `runs/dense_retriever/`, `runs/train_reader/`, `runs/train_ranker/`, `runs/validate_reader/`, `runs/validate_endtoend/` to see details.