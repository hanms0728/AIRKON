import numpy as np


def parallelogram_from_triangle(p0, p1, p2):
    """Restores four vertices of a parallelogram using three triangle points."""
    p3 = 2 * p0 - p1
    p4 = 2 * p0 - p2
    return np.stack([p1, p2, p3, p4], axis=0)


def aabb_of_poly4(poly4):
    """Returns the axis-aligned bbox (x0, y0, w, h) for a 4-point polygon."""
    xs = poly4[:, 0]
    ys = poly4[:, 1]
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def iou_aabb_xywh(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-9
    return inter / union
