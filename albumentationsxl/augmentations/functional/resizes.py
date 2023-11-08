import pyvips
from ...core.transforms_interface import (
    KeypointInternalType,
)

__all__ = ["keypoint_scale", "resize", "scale", "smallest_max_size", "longest_max_size", "py3round"]


def keypoint_scale(keypoint: KeypointInternalType, scale_x: float, scale_y: float) -> KeypointInternalType:
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    return x * scale_x, y * scale_y, angle, scale * max(scale_x, scale_y)


def resize(img: pyvips.Image, height: int, width: int, interpolation: int = pyvips.enums.Kernel.LINEAR):
    img_height, img_width = img.height, img.width
    if height == img_height and width == img_width:
        return img

    # compute scales
    hscale = width / img_width
    vscale = height / img_height

    return img.resize(hscale, vscale=vscale, kernel=interpolation)


def scale(img: pyvips.Image, scale: float, interpolation: int = pyvips.enums.Kernel.LINEAR) -> pyvips.Image:
    """resize in cv2 and pyvips are opposites in how they work.
    pyvips resizes using a scale factor, while cv2 accepts new widths/heights
    """
    return img.resize(scale, kernel=interpolation)


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def _func_max_size(img: pyvips.Image, max_size, interpolation, func):
    height, width = img.height, img.width

    scale = max_size / float(func(width, height))

    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        img = resize(img, height=new_height, width=new_width, interpolation=interpolation)
    return img


def longest_max_size(img: pyvips.Image, max_size: int, interpolation: str) -> pyvips.Image:
    return _func_max_size(img, max_size, interpolation, max)


def smallest_max_size(img: pyvips.Image, max_size: int, interpolation: str) -> pyvips.Image:
    return _func_max_size(img, max_size, interpolation, min)
