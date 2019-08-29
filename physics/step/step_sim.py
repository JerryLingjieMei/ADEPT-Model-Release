import pybullet as p
from .lite_step_objects import LiteObjectStepManager
from utils.constants import CONTENT_FOLDER
import os

_STEP_TIME = 4
_SIM_TIME_STEP = 0.01


def step(objects, forward=True):
    """Step a simulation"""
    om = LiteObjectStepManager(objects, os.path.join(CONTENT_FOLDER, "physics/data"), forward=forward)
    p.setTimeStep(_SIM_TIME_STEP)
    for i in range(_STEP_TIME):
        p.stepSimulation()
    new_objects = []
    for object in om.object_ids:
        new_objects.append(om.get_object_motion(object))
    return new_objects


def reverse_step(config):
    """Reverse step a simulation"""
    return step(config, False)
