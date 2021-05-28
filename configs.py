D1 = 50
D2 = 800

BATCH_SIZE = 8
TTS        = 0.2
SEED       = 42
EPOCHS     = 35
SHIFT      = 1
STACKS     = 2
INTERVAL   = 0.5

LEARNING_RATE = 0.0001

TENSORBOARD = True
DELETE_LOGS = True

LOADING_BAR_FOLD = 64

MODEL_SAVE_DIR = 'models/model.tf'

DATA_FILE = 'data/save.txt'

# 'roll' or 'shuffle' or None
SAMPLE_TYPE = 'shuffle'
assert (SAMPLE_TYPE == 'roll') or (SAMPLE_TYPE == 'shuffle') or (SAMPLE_TYPE is None)
