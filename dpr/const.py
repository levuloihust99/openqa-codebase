# global
TRAIN_DATA_SIZE = 58806
EPOCHS = 40
BATCH_SIZE = 16

# AdamW
LEARNING_RATE = 1e-5
ADAM_EPS = 1e-8
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 100

# Gradient clipping
MAX_GRAD_NORM = 1.0

# training
CTX_ENCODER_TRAINABLE = True
QUESTION_ENCODER_TRAINABLE = True
CHECKPOINT_PATH = "gs://levuloi-dense-passage-retriever/checkpoints/retriever"
DATA_PATH = "gs://levuloi-dense-passage-retriever/data/retriever/N5000-INT/*"
SHUFFLE = True
SHUFFLE_SEED = 123
MAX_CONTEXT_LENGTH = 256
MAX_QUERY_LENGTH = 256