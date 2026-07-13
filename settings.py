DEVICE = 'cuda:0'

SEED = 123

MODEL_NAME = 'bagnet17'
NUM_CLASSES = 5

TRAIN_DIR = 'data/stanford_dogs_small/train'
TEST_DIR = 'data/stanford_dogs_small/test'
# TRAIN_DIR = 'data/CUB_200_2011/dataset/train'
# TEST_DIR = 'data/CUB_200_2011/dataset/test'
# TEST_DIR = 'data/CUB_200_2011/dataset/test_crop'

DATA_PERCENTAGE = 1.0
VAL_SPLIT = 0.2

BATCH_SIZE = 16

NUM_EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 0

BASE_MODEL_BIAS = True
MODEL_PATH = 'models/'
SAVE_MIN_ACC = 0.5

IMG_SIZE = 224

IMG_MEAN = [0.4761, 0.4518, 0.3910]
IMG_STD  = [0.2580, 0.2525, 0.2571]

TENSORBOARD_LOGDIR_PREFIX = 'runs/'
