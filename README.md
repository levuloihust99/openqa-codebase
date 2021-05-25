# Requirements
This code base is tested on:
- Python 3.8.5
- Tensorflow 2.4.1
- Transformers 4.4.2

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

# Run code
```shell
$ python train_retriever.py
```

# Configuration
All configurations are placed at `dpr/config.py`

# Data and checkpoints
Training data, model checkpoints and other resources are stored in Google Cloud Storage, at `gs://openqa-dpr`. If you want to use these resources, please email me via email address `levuloihust@outlook.com`
