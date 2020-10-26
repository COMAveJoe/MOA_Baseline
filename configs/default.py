# author: yx
# date: 2020/10/16 11:20

from yacs.config import CfgNode as CN

# ----------------------------------------------------------------------------------------------------------------------
# Convention about Training/ Test specific parameters
# ----------------------------------------------------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing,
# the corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.

# ----------------------------------------------------------------------------------------------------------------------
# Config definition
# ----------------------------------------------------------------------------------------------------------------------

_C = CN()

# ----------------------------------------------------------------------------------------------------------------------
# Model basic parameters
# ----------------------------------------------------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cpu'
_C.MODEL.META_ARCHITECTURE = 'MOA_MLPv2'
_C.MODEL.NUM_CATS = [2 + 1, 3 + 1, 2 + 1]
_C.MODEL.CATS_EMB_SIZE = [1, 1, 1]
_C.MODEL.AUX = 330
_C.MODEL.NUM_NUMERICALS = 872
_C.MODEL.NUM_CLASS = 206
_C.MODEL.HIDDEN_SIZE = 2048


# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = r'D:\YX\resource\lish-moa\train_features.csv'
_C.DATASET.TRAIN_TARGETS_SCORED = r'D:\YX\resource\lish-moa\train_targets_scored.csv'
_C.DATASET.TRAIN_TARGETS_NON_SCORED = r'D:\YX\resource\lish-moa\train_targets_nonscored.csv'

_C.DATASET.TEST = r'D:\YX\resource\lish-moa\test_features.csv'

_C.DATASET.SUBMISSION = r'D:\YX\resource\lish-moa\sample_submission.csv'

_C.DATASET.NORMALIZE = True
_C.DATASET.REMOVE_VEHICLE = True

# ----------------------------------------------------------------------------------------------------------------------
# DataLoader
# ----------------------------------------------------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 1

# ----------------------------------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 1.0e-3
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.ALPHA = 0.5

_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.CHECKPOINT_PERIOD = 5
_C.SOLVER.LOG_PERIOD = 4

_C.SOLVER.BATCH_SIZE = 128

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128
_C.TEST.WEIGHT = r"D:\YX\code\MOA_Baseline-master\MOA_Baseline-master\output\moa_mlp_2_checkpoint_8400.pt"


# ----------------------------------------------------------------------------------------------------------------------
# Misc options
# ----------------------------------------------------------------------------------------------------------------------
_C.OUTPUT_DIR = r"D:\YX\code\MOA_Baseline-master\MOA_Baseline-master\output"
_C.INFERENCE_RESULT_DIR = r'D:\YX\code\MOA_Baseline-master\MOA_Baseline-master\infer_result'
