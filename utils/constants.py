import os
import random

CONTENT_FOLDER = "/root/adept-model/"

WIDTH = 480
HEIGHT = 320

AREA_MIN_THRESHOLD = 50

# Sphere, Cube, Cylinder, Cones are not occluders
CATEGORY2ID = {
    "Sphere": 1,
    "Cube": 1,
    "Cylinder": 1,
    "Cone": 1,
    "Occluder": 2
}

TERMS = ["type", "color", "location", "velocity", "rotation", "scale"]
COLORS = ["red", "blue", "green", "brown", "purple", "cyan", "yellow"]
TYPES = ["BG", "Sphere", "Occluder"]

SHAPES2TYPES = {
    'Occluder': 'cube',
    'Sphere': 'sphere',
}

COLORS2RGB = {
    "red": [173, 35, 35],
    "blue": [42, 75, 215],
    "green": [29, 105, 20],
    "brown": [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan": [41, 208, 208],
    "yellow": [255, 238, 51]
}
