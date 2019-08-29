from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "PARTICLE_FILTER"
# Particles to be used in the particle filter
_C.MODEL.N_PARTICLES = 128
# Threshold for minimal area for an objects to be considered visible
_C.MODEL.AREA_THRESHOLD = 200.


# -----------------------------------------------------------------------------
# Dynamics Model
# -----------------------------------------------------------------------------
_C.MODEL.STEP = CN()
_C.MODEL.STEP.PERTURBATION = CN()
# Whether to perturb the objects
_C.MODEL.STEP.PERTURBATION.TO_PERTURB = True
# Sigma in the velocity term
_C.MODEL.STEP.PERTURBATION.VELOCITY_SIGMA = [.01, .06]
_C.MODEL.STEP.PERTURBATION.SCALE_SIGMA = .0005
# Sigma in the location term
_C.MODEL.STEP.PERTURBATION.LOCATION_SIGMA = [.01, .06]
# Sigma in the velocity term, multiplicative
_C.MODEL.STEP.PERTURBATION.VELOCITY_LAMBDA = [.01, .06]

# -----------------------------------------------------------------------------
# Magic in the dynamics model
# -----------------------------------------------------------------------------
_C.MODEL.STEP.MAGIC = CN()
# Whether to use magic
_C.MODEL.STEP.MAGIC.USE_MAGIC = True
# The probability to disappear
_C.MODEL.STEP.MAGIC.DISAPPEAR_PROBABILITY = .02
# The penalty for magically disappearing
_C.MODEL.STEP.MAGIC.DISAPPEAR_PENALTY = 10.
# The probability for magically stopping
_C.MODEL.STEP.MAGIC.STOP_PROBABILITY = .02
# The penalty for magically stopping
_C.MODEL.STEP.MAGIC.STOP_PENALTY = 1.
# The probability for magically accelerating
_C.MODEL.STEP.MAGIC.ACCELERATE_PROBABILITY = .04
# The penalty for magically accelerating
_C.MODEL.STEP.MAGIC.ACCELERATE_PENALTY = 1.
# The magnitude for magically accelerating
_C.MODEL.STEP.MAGIC.ACCELERATE_LAMBDA = 1.5

# -----------------------------------------------------------------------------
# Particle filter
# -----------------------------------------------------------------------------
# The period for particle filter to resample
_C.MODEL.RESAMPLE = CN()
# Resample every period
_C.MODEL.RESAMPLE.PERIOD = 1
# Scaling on nll
_C.MODEL.RESAMPLE.FACTOR = 1.


# -----------------------------------------------------------------------------
# Mass sampler
# -----------------------------------------------------------------------------
_C.MODEL.MASS = CN()
# Whether to sample mass
_C.MODEL.MASS.TO_SAMPLE_MASS = False
# The log mean of mass
_C.MODEL.MASS.LOG_MASS_MU = 0
# The log stdev of mass
_C.MODEL.MASS.LOG_MASS_SIGMA = 1

# -----------------------------------------------------------------------------
# Observation Model
# -----------------------------------------------------------------------------
_C.MODEL.UPDATING = CN()
_C.MODEL.UPDATING.MATCHED = CN()
# Loss for matched object updating
_C.MODEL.UPDATING.MATCHED.LOSS = "Smoothed_L_Half"
# Sigma in the location term
_C.MODEL.UPDATING.MATCHED.LOCATION_SIGMA = .2
# Sigma in the velocity term
_C.MODEL.UPDATING.MATCHED.VELOCITY_SIGMA = .2
_C.MODEL.UPDATING.MATCHED.SCALE_SIGMA = .05

_C.MODEL.UPDATING.UNMATCHED_BELIEF = CN()
# Base Penalty coefficient for unseen object
_C.MODEL.UPDATING.UNMATCHED_BELIEF.BASE_PENALTY = 1.
# Penalty coefficient for unseen object w.r.t. mask area shown
_C.MODEL.UPDATING.UNMATCHED_BELIEF.MASK_PENALTY = .0001

_C.MODEL.UPDATING.UNMATCHED_OBSERVATION = CN()
# Penalty for object appearing
_C.MODEL.UPDATING.UNMATCHED_OBSERVATION.PENALTY = .02
_C.MODEL.UPDATING.UNMATCHED_OBSERVATION.MAX_PENALTY = 12.

_C.MODEL.MATCHER = CN()
# PENALTY FOR MISMATCHED OBJECT TYPES, ONLY BETWEEN OCCLUDER AND OTHER
_C.MODEL.MATCHER.TYPE_PENALTY = 10.
# PENALTY FOR MISMATCHED OBJECT COLOR
_C.MODEL.MATCHER.COLOR_PENALTY = 12.
# PENALTY FOR MISMATCHED OBJECT WHEN THEY ARE AFAR
_C.MODEL.MATCHER.DISTANCE_PENALTY = 20.
# THE THRESHOLD FOR OBJECT BEING AFAR
_C.MODEL.MATCHER.DISTANCE_THRESHOLD = 2.
# THE BASE PENALTY BETWEEN PLACEHOLDER AND OBJECTS
_C.MODEL.MATCHER.BASE_PENALTY = 8.

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.CASE_NAMES = []
_C.USE_GT_OBSERVATION = False
_C.ANNOTATION_FOLDER = ""
_C.OBSERVATION_FOLDER = ""
_C.OUTPUT_FOLDER = ""
_C.LOG_PREFIX = ""
_C.PLOT_SUMMARY = True
_C.ID = 0
