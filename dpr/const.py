# global
TRAIN_DATA_SIZE = 60000
EPOCHS = 40
BATCH_SIZE = 8

# AdamW
LEARNING_RATE = 2e-5
ADAM_EPS = 1e-8
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 100

# Gradient clipping
MAX_GRAD_NORM = 2.0

# training
CTX_ENCODER_TRAINABLE = True
QUESTION_ENCODER_TRAINABLE = True
CHECKPOINT_PATH = "gs://dpr/checkpoints/retriever/V3"
DATA_PATH = "gs://dpr/data/retriever/V3/N5000-INT"
SHUFFLE = True
SHUFFLE_SEED = 123
MAX_CONTEXT_LENGTH = 256
MAX_QUERY_LENGTH = 256