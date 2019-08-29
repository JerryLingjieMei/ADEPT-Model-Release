from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Meta architecture for the model
_C.MODEL.META_ARCHITECTURE = "DERENDER"
# Input channels for the model
_C.MODEL.IN_CHANNELS = 12
# The size of pooling kernel in the last layer of derender
_C.MODEL.POOLING_KERNEL_SIZE = (10, 15)
# Number of derender visual feature channels
_C.MODEL.FEATURE_CHANNELS = 512
# Number of intermediate layer channels
_C.MODEL.MID_CHANNELS = 128
# Number of output channels
_C.MODEL.OUT_CHANNELS = 3

# -----------------------------------------------------------------------------
# Attribute representation
# -----------------------------------------------------------------------------
_C.MODEL.ATTRIBUTES = CN()
# Object attribute representation
_C.MODEL.ATTRIBUTES.NAME = "BASE_ATTRIBUTES"
# Number of object classes, including background
_C.MODEL.ATTRIBUTES.N_TYPES = 3
# Number of colors
_C.MODEL.ATTRIBUTES.N_COLORS = 7


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.MODEL.LOSS = CN()
# Choice of loss
_C.MODEL.LOSS.NAME = "L2"
_C.MODEL.LOSS.GRADIENT_LAMBDA = .1
_C.MODEL.LOSS.BCE_LAMBDA = .002

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""

_C.DATASETS.VAL = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

# Checkpoint every _ iterations
_C.SOLVER.CHECKPOINT_PERIOD = 1000
# Run validation every _ iterations
_C.SOLVER.VALIDATION_PERIOD = 1000
# Validate _ batches every time
_C.SOLVER.VALIDATION_LIMIT = 100

_C.SOLVER.IMS_PER_BATCH = 8

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "."
_C.VISUALIZATION = ""
