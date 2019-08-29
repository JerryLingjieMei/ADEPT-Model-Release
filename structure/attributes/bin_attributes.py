from torch import nn
import torch
import torchvision
import torch.nn.functional as F
from math import pi


class BinAttributes(object):

    @staticmethod
    def cum_sum(sequence):
        r, s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    @staticmethod
    def rotation2bins(rotation, n_bins):
        return torch.stack([F.one_hot(((rotation[:, 0] + pi / 2) / pi * n_bins).long(), n_bins),
                            F.one_hot(((rotation[:, 1] + pi) / pi / 2 * n_bins).long(), n_bins),
                            F.one_hot(((rotation[:, 2] + pi) / pi / 2 * n_bins).long(), n_bins)], dim=1)

    @staticmethod
    def bins2rotation(bins, n_bins):
        x = torch.arange((n_bins,), dtype=float, device="cuda")
        return torch.stack([torch.sum(bins[:, :n_bins] * x, dim=1) * pi / n_bins - pi / 2,
                            torch.sum(bins[:, :n_bins] * x, dim=1) * 2 * pi / n_bins - pi,
                            torch.sum(bins[:, :n_bins] * x, dim=1) * 2 * pi / n_bins - pi], dim=1)

    def __init__(self, attribute_cfg):
        super(BinAttributes, self).__init__()
        self.n_types = attribute_cfg.N_TYPES
        self.n_colors = attribute_cfg.N_COLORS
        self.n_bins = attribute_cfg.N_BINS
        self.keys = ["type", "color", "location", "velocity", "rotation", "scale"]
        self.value_lengths = [self.n_types, self.n_colors, 3, 3, 3 * self.n_bins, 3]
        self.value_indices = self.cum_sum(self.value_lengths)

    def __len__(self):
        return self.value_indices[-1]

    def forward(self, input):
        x = torch.Tensor((input.shape[0], self.value_indices[-1]), device="cuda")
        for i, key in enumerate(self.keys):
            if key == "rotation":
                x[:, self.value_indices[i]:self.value_indices[i + 1]] = self.rotation2bins(input[key])
            else:
                x[:, self.value_indices[i]:self.value_indices[i + 1]] = input[key]
        return x

    def backward(self, input):
        x = {}
        for i, key in enumerate(self.keys):
            if key == "rotation":
                x[key] = self.bins2rotation(input[:, self.value_indices[i]:self.value_indices[i + 1]])
            else:
                x[key] = input[:, self.value_indices[i]:self.value_indices[i + 1]]
        return x
