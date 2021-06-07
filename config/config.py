import os
from yacs.config import CfgNode as CN
from .paths_catalogue import DatasetsCatalog

_C = CN()

# Options for the choice of data
_C.DATASET = CN()

_C.DATASET.DATASET = 'DatasetName'
# If you want to override dataset path set this option
# otherwise path will be set according to paths catalogue
_C.DATASET.PATH = ''


# Model general options
_C.MODEL = CN()

# Here usually the name / identifier of the model
_C.MODEL.MODEL = 'Stage1Base'
# Initial mode of the model
_C.MODEL.MODE = 'train'
# Dataloader shuffling (active only for train split)
_C.MODEL.SHUFFLE = True
# Optimiser
_C.MODEL.OPTIMISER = 'Adam'
# Learning rate
_C.MODEL.LEARNING_RATE = 0.001
_C.MODEL.WEIGHT_DECAY = 5e-4

'''
I suggest adding model relevant options here
'''

# Loss related group
_C.MODEL.LOSS = CN()


# Logging
_C.LOGGING = CN()
# Logging dir - directory used for logging (will be appended with timestamp)
_C.LOGGING.LOGDIR = '../outputs/_logtest'
# Use of tensorboard (will be placed in logdir)
_C.LOGGING.TENSORBOARD = True
# Debug log presence
_C.LOGGING.DEBUG = True
# Debug message frequency (in iters)
_C.LOGGING.DEBUG_MSG = 10
# Display message frequency (in iters)
_C.LOGGING.DISPLAY = 20


# Other config options
_C.CONFIG = CN()
# Number of workers for loader
_C.CONFIG.NUM_WORKERS = 8
# Batch size
_C.CONFIG.BATCH = 256
_C.CONFIG.BATCH_VAL = 256
# Number of epochs
_C.CONFIG.EPOCHS = 100
# Checkpoint frequency (in number of iterations - may be better than epochs
# cause of better control) - or it probably does not matter tbh
_C.CONFIG.CHECKPOINT = 500
# Zeroing epoch and iteration counters, e.g. next stage of training
_C.CONFIG.ZERO_TRAINING = False
# Load path
_C.CONFIG.CHECKPOINT_PATH = ''
# Use of tensorboard
_C.CONFIG.TENSORBOARD = True
# Test split
_C.CONFIG.TEST_SPLIT = 'test'


# Some opts I used for debugging
_C.DEBUG = CN()
_C.DEBUG.TRAIN_SPLIT = 'train'
_C.DEBUG.TRAIN_LEN = 0
_C.DEBUG.VAL_SPLIT = 'val'
_C.DEBUG.VAL_LEN = 0


# Info options
_C.INFO = CN()
# Empty field to keep track of config file
_C.INFO.CFG_PATH = ''
# Empty field to keep track of project dir (usually ..)
_C.INFO.PROJECT_DIR = '..'
# Comment
_C.INFO.COMMENT = 'Default comment'


def get_cfg(config_file):
    # We don't touch _C as we may want to check defaults for w/e reason
    cfg = _C.clone()
    if config_file.lower() in ['default', 'defaults']:
        print('Returning default configuration')
    else:
        if os.path.isfile(config_file):
            cfg.merge_from_file(config_file)
        else:
            print('Incorrect path provided, loading defaults')

    cfg = fill_catalogue_paths(cfg)
    cfg.merge_from_list(['INFO.CFG_PATH', os.path.abspath(config_file)])

    return cfg


# Here we fill from catalouge of paths, you can add more relevant paths
def fill_catalogue_paths(cfg):
    if cfg.DATASET.PATH == '':
        path = DatasetsCatalog.get(cfg.DATASET.DATASET)
        cfg.merge_from_list(["DATASET.PATH", path])

    return cfg
