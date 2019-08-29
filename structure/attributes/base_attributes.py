from torch import nn
import torch
import torchvision


class BaseAttributes(object):

    @staticmethod
    def cum_sum(sequence):
        r, s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, attribute_cfg):
        super(BaseAttributes, self).__init__()
        self.n_types = attribute_cfg.N_TYPES
        self.n_colors = attribute_cfg.N_COLORS
        self.term = ["type", "color", "location", "velocity", "rotation", "scale"]
        self.loss_methods = {k: nn.MSELoss() for k in self.term}
        self.value_lengths = [self.n_types, self.n_colors, 3, 3, 3, 3]
        self.value_indices = self.cum_sum(self.value_lengths)

    def __len__(self):
        return self.value_indices[-1]

    def forward(self, input):
        x = torch.zeros((input[self.term[0]].shape[0], self.value_indices[-1])).to(input.device)
        for i, term in enumerate(self.term):
            x[:, self.value_indices[i]:self.value_indices[i + 1]] = input[term]
        return x

    def backward(self, input):
        x = {}
        for i, term in enumerate(self.term):
            x[term] = input[:, self.value_indices[i]:self.value_indices[i + 1]]
        return x

    def loss(self, input, target):
        loss_dict = {}
        for term, loss_method in self.loss_methods.items():
            loss = loss_method(input[term], target[term])
            loss_dict[term] = loss
        loss_dict["loss"] = sum(l for l in loss_dict.values())
        return loss_dict

