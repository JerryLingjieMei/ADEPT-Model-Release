def iou(m1, m2):
    intersect = m1 * m2
    union = 1 - (1 - m1) * (1 - m2)
    return intersect.sum() / union.sum()


def reverse_xyz(t):
    """Point an 3d vector to the opposite direction"""
    return [-t[0], -t[1], -t[2]]


def reverse_euler(t):
    """Point a xyz euler to the opposite direction"""
    return [-t[2], -t[1], -t[0]]
