import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34

from structure.attributes import build_attributes


class Derender(nn.Module):

    def __init__(self, cfg):
        super(Derender, self).__init__()
        self.attributes = build_attributes(cfg.MODEL.ATTRIBUTES)

        resnet = resnet34(pretrained=True)
        resnet_layers = list(resnet.children())

        resnet_layers.pop()
        resnet_layers.pop(0)
        resnet_layers.insert(0, nn.Conv2d(cfg.MODEL.IN_CHANNELS, 64, kernel_size=3, stride=2, padding=1))
        resnet_layers[-1] = nn.AvgPool2d(kernel_size=cfg.MODEL.POOLING_KERNEL_SIZE)

        self.backbone = nn.Sequential(*resnet_layers)
        self.middle_layer = nn.Linear(cfg.MODEL.FEATURE_CHANNELS, cfg.MODEL.MID_CHANNELS)
        self.final_layer = nn.Linear(cfg.MODEL.MID_CHANNELS, len(self.attributes))

        self.loss = self.attributes.loss

    def forward(self, inputs):
        img_tuple = inputs["img_tuple"]
        x = self.backbone(img_tuple)
        x = x.view(x.size(0), -1)
        x = F.relu(self.middle_layer(x))
        x = self.final_layer(x)
        output = self.attributes.backward(x)
        if self.training:
            loss_dict = self.loss(output, inputs["attributes"])
            return {"output": output, "loss_dict": loss_dict}
        return {"output": output, "loss_dict": None}
