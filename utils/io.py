import yaml
import json
import os
from easydict import EasyDict
from PIL import Image


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def read_serialized(file_name):
    """Read json and yaml file"""
    if file_name is not None:
        with open(file_name, "r") as f:
            if file_name.endswith(".json"):
                x = json.load(f)
            elif file_name.endswith(".yaml"):
                x = yaml.full_load(f)
            else:
                raise FileNotFoundError
    if isinstance(x, dict):
        return EasyDict(x)
    else:
        return x


def write_serialized(var, file_name):
    """Write json and yaml file"""
    assert file_name is not None
    with open(file_name, "w") as f:
        if file_name.endswith(".json"):
            json.dump(var, f, indent=4)
        elif file_name.endswith(".yaml"):
            yaml.safe_dump(var, f, indent=4)
        else:
            raise FileNotFoundError
