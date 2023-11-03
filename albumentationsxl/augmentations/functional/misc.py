import pyvips
import numpy as np


__all__ = ["bbox_from_mask"]


def bbox_from_mask(mask: pyvips.Image):
    """Create bounding box from binary mask (fast version)

    Args:
        mask (numpy.ndarray): binary mask.

    Returns:
        tuple: A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    """
    rows = np.any(mask, axis=1)
    if not rows.any():
        return -1, -1, -1, -1
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max + 1, y_max + 1
