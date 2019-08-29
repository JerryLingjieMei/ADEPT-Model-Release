from copy import copy
from math import pi

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
from easydict import EasyDict

from physics.visualize.mask_visualize import visible_image
from utils.constants import COLORS, TYPES, TERMS
from utils.misc import to_cpu, FONT


class AttributeReconstructionVisualizer(object):
    _default_camera = EasyDict({"camera_look_at": [-1.5, 0, 0],
                                "camera_phi": 0,
                                "camera_rho": 7.2,
                                "camera_theta": 20})

    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    @staticmethod
    def visualize_attribute(attribute, display_size):
        text = ""
        for term in TERMS:
            if term == "type":
                t = TYPES[torch.argmax(attribute[term]).item()]
                text += "{:<10}: {:}\n".format(term, t)
            elif term == "color":
                c = COLORS[torch.argmax(attribute[term]).item()]
                text += "{:<10}: {:}\n".format(term, c)
            elif term == "rotation":
                x, y, z = attribute[term] * 180 / pi
                text += "{:<10}: {:<8.2f} {:<8.2f} {:<8.2f}\n".format(term, x, y, z)
            else:
                x, y, z = attribute[term]
                text += "{:<10}: {:<8.2f} {:<8.2f} {:<8.2f}\n".format(term, x, y, z)
        img = Image.new("RGB", (display_size[1], display_size[0]))
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(FONT, 16)
        d.text((16, 16), text, font=f)
        return TF.to_tensor(img)

    @staticmethod
    def visualize_reconstruction(attribute):
        a = copy(attribute)
        a["type"] = TYPES[torch.argmax(a["type"]).item()]
        a["color"] = COLORS[torch.argmax(a["color"]).item()]
        if a["type"] == "BG":
            return TF.to_tensor(visible_image([], AttributeReconstructionVisualizer._default_camera))
        return TF.to_tensor(visible_image([a], AttributeReconstructionVisualizer._default_camera))

    def visualize(self, inputs, outputs, iteration):
        summary_images = []

        inputs = to_cpu(inputs)
        outputs = to_cpu(outputs)
        for i in range(min(4, inputs["img_tuple"].shape[0])):
            target_image = inputs["img_tuple"][i, 9:]
            target_segm = inputs["img_tuple"][i, :3]
            target_attribute = {k: v[i] for k, v in inputs["attributes"].items()}
            output_attribute = {k: v[i] for k, v in outputs["output"].items()}

            target_ann = self.visualize_attribute(target_attribute, target_image.shape[1:])
            output_ann = self.visualize_attribute(output_attribute, target_image.shape[1:])

            p.connect(p.DIRECT)
            target_reconstruction = self.visualize_reconstruction(target_attribute)
            output_reconstruction = self.visualize_reconstruction(output_attribute)
            p.disconnect()

            summary_image = torch.cat(
                [torch.cat([target_image, target_reconstruction, output_reconstruction], dim=2),
                 torch.cat([target_segm, target_ann, output_ann], dim=2)], dim=1)
            summary_images.append(summary_image)

        self.summary_writer.add_image("reconstruction", make_grid(summary_images, nrow=2), iteration)
