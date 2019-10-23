import random
import os
import glob

import socket

import numpy as np
import torch
from skimage import measure


def assert_proper_output_dir(f, output_dir):
    assert os.path.splitext(os.path.basename(f))[0] == os.path.basename(
        output_dir), "Output directory should be {}".format(os.path.splitext(os.path.basename(f))[0])


def to_cuda(x):
    if torch.is_tensor(x):
        return x.to("cuda")
    elif isinstance(x, list):
        return [to_cuda(_) for _ in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if torch.is_tensor(x):
        return x.to("cpu")
    elif isinstance(x, list):
        return [to_cpu(_) for _ in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x


def to_torch(x):
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    elif isinstance(x, list):
        if isinstance(x[0], float) or isinstance(x[0], int):
            return torch.Tensor(x)
        else:
            return [to_torch(_) for _ in x]
    return x


def gather_loss_dict(outputs):
    outputs["loss_dict"] = {k: v.mean() for k, v in outputs["loss_dict"].items()}
    return outputs["loss_dict"]


def rand():
    return random.random()


def get_font():
    font_library = "/usr/share/fonts/truetype"
    if os.path.exists(os.path.join(font_library, "ttf-bitstream-vera", "VeraMono.ttf")):
        return os.path.join(font_library, "ttf-bitstream-vera", "VeraMono.ttf")
    for font_folder in glob.glob(os.path.join(font_library, "*")):
        for font in glob.glob(os.path.join(font_folder, "*Mono.ttf")):
            return font
    for font_folder in glob.glob(os.path.join(font_library, "*")):
        for font in glob.glob(os.path.join(font_folder, "*Regular.ttf")):
            return font


FONT = get_font()


def get_host_id():
    return 0
    # return int(socket.gethostname()[-2:]) - 1


def mask2contour(mask):
    contours = measure.find_contours(mask, .5)
    segmentation = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation.append(contour.ravel().tolist())
    return segmentation