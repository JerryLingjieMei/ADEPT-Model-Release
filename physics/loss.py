import math


def l1(a, b, std):
    return sum(abs(x - y) for x, y in zip(a, b)) / std


def l2(a, b, std):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / (std ** 2)


def smoothed_l1(a, b, std):
    c = (abs(x - y) / std for x, y in zip(a, b))
    return sum(z if z > 1 else z ** 2 for z in c)


def smoothed_l_half(a, b, std):
    c = (abs(x - y) / std for x, y in zip(a, b))
    return sum(math.sqrt(z) if z > 1 else z ** 2 for z in c)


LOSS_MAP = {
    "L1": l1,
    "L2": l2,
    "Smoothed_L1": smoothed_l1,
    "Smoothed_L_Half": smoothed_l_half
}


def build_loss(name):
    return LOSS_MAP[name]
