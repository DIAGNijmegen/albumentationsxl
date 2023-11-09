"""

Functional representations of the augmentations. Most will function as functional pyvips aliases.

"""
import cv2
import math
import pyvips
import numpy as np
import skimage.transform

from collections.abc import Sequence

from albumentationsxl.augmentations.utils import (
    angle_2pi_range,
)
from ...core.bbox_utils import denormalize_bbox, normalize_bbox
from ...core.transforms_interface import (
    BoxInternalType,
    ImageColorType,
    KeypointInternalType,
)

__all__ = [
    "bbox_flip",
    "bbox_hflip",
    "bbox_rot90",
    "bbox_rotate",
    "bbox_transpose",
    "bbox_vflip",
    "elastic_transform",
    "hflip",
    "keypoint_flip",
    "keypoint_hflip",
    "keypoint_rot90",
    "keypoint_rotate",
    "keypoint_transpose",
    "keypoint_vflip",
    "pad_with_params",
    "random_flip",
    "rot90",
    "rotate",
    "transpose",
    "vflip",
    "warp_affine",
    "rotation2DMatrixToEulerAngles",
    "keypoint_affine",
    "bbox_affine",
]


def bbox_rot90(bbox: BoxInternalType, factor: int, rows: int, cols: int) -> BoxInternalType:  # skipcq: PYL-W0613
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox: A bounding box tuple (x_min, y_min, x_max, y_max).
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    x_min, y_min, x_max, y_max = bbox[:4]
    if factor == 1:
        bbox = y_min, 1 - x_max, y_max, 1 - x_min
    elif factor == 2:
        bbox = 1 - x_max, 1 - y_max, 1 - x_min, 1 - y_min
    elif factor == 3:
        bbox = 1 - y_max, x_min, 1 - y_min, x_max
    return bbox


@angle_2pi_range
def keypoint_rot90(keypoint: KeypointInternalType, factor: int, rows: int, cols: int, **params) -> KeypointInternalType:
    """Rotates a keypoint by 90 degrees CCW (see np.rot90)

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        factor: Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if factor not in set {0, 1, 2, 3}

    """
    x, y, angle, scale = keypoint[:4]

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if factor == 1:
        x, y, angle = y, (cols - 1) - x, angle - math.pi / 2
    elif factor == 2:
        x, y, angle = (cols - 1) - x, (rows - 1) - y, angle - math.pi
    elif factor == 3:
        x, y, angle = (rows - 1) - y, x, angle + math.pi / 2

    return x, y, angle, scale


def rotate(
    img: pyvips.Image,
    angle: float,
    interp: pyvips.Interpolate | str,
    background: int | float | Sequence[int] | Sequence[float],
):
    return img.similarity(angle=angle, interpolate=interp, background=background)


def bbox_rotate(bbox: BoxInternalType, angle: float, method: str, rows: int, cols: int) -> BoxInternalType:
    """Rotates a bounding box by angle degrees.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        angle: Angle of rotation in degrees.
        method: Rotation method used. Should be one of: "largest_box", "ellipse". Default: "largest_box".
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    References:
        https://arxiv.org/abs/2109.13488

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    if method == "largest_box":
        x = np.array([x_min, x_max, x_max, x_min]) - 0.5
        y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    elif method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        data = np.arange(0, 360, dtype=np.float32)
        x = w * np.sin(np.radians(data)) + (w + x_min - 0.5)
        y = h * np.cos(np.radians(data)) + (h + y_min - 0.5)
    else:
        raise ValueError(f"Method {method} is not a valid rotation method.")
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5

    x_min, x_max = min(x_t), max(x_t)
    y_min, y_max = min(y_t), max(y_t)

    return x_min, y_min, x_max, y_max


@angle_2pi_range
def keypoint_rotate(keypoint, angle, rows, cols, **params):
    """Rotate a keypoint by angle.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        angle (float): Rotation angle.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    center = (cols - 1) * 0.5, (rows - 1) * 0.5
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return x, y, a + math.radians(angle), s


def vflip(img: pyvips.Image) -> pyvips.Image:
    return img.flipver()


def hflip(img: pyvips.Image) -> pyvips.Image:
    return img.fliphor()


def rot90(img: pyvips.Image, factor: int) -> pyvips.Image:
    """

    Parameters
    ----------
    img : pyvips.Image
        The pyvips image as uchar (uint8) dtype
    factor : int
        Number of times the input will be rotated by 90 degrees.
    Returns
    -------
    img: pyvips.Image

    """
    # only 4 choices for 90 degrees, regardless of the factor value
    factor = factor % 4
    options = [img, img.rot90, img.rot180, img.rot270]
    print(factor)
    result = options[factor] if factor == 0 else options[factor]()
    return result


def random_flip(img: pyvips.Image, code: int) -> pyvips.Image:
    """Pyvips random flips that emulated cv2.flip

    cv2.flip(img,d) takes an additional code that specifies the flip:
    -1: both horizontal and vertical flip, 0 is vertical, 1 is horizontal


    Parameters
    ----------
    img : pyvips.Image
        The pyvips image as uchar (uint8) dtype
    code : int
        Mode for flipping the image, follows cv2.flip() coding
    Returns
    -------
    img: pyvips.Image

    """
    if code == 1:
        img = img.fliphor()
    elif code == 0:
        img = img.flipver()
    else:
        img = img.fliphor().flipver()

    return img


def transpose(img: pyvips.Image) -> pyvips.Image:
    """Albumentation transpose an image

    Albumentations transposes by switching rows and column in numpy. This is not the same as the matrix transpose
    used in linear algebra. Instead, the outcome is a 270 degree flip. For a proper transpose, additionally
    flip the result vertically.

    Parameters
    ----------
    img: pyvips.Image
        The pyvips image as uchar (uint8) dtype
    Returns
    -------
    img: pyvips.Image

    """

    return img.rot270()


def elastic_transform(
    img: pyvips.Image,
    alpha: float = 1.0,
    sigma: float = 50.0,
    interpolation: str | pyvips.Interpolate = pyvips.Interpolate.new("bilinear"),
    background: int | float | Sequence[int] | Sequence[float] = 255,
    same_dxdy: bool = False,
) -> pyvips.Image:
    """Apply elastic transformation on the image

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Parameters
    ----------
    img : Pyvips Image object
        The image on which to apply elastic deformation
    sigma : int
        Elasticity coefficient : standard deviation for Gaussian blur, controls smoothness, higher is smoother
    alpha : int
        Scaling  factor that controls the intensity (magnitude) of the displacements
    interpolation : pyvips.Interpolate
        The interpolation to use, can be one of bilinear, cubic, linear, nearest
    same_dx_dy : bool
        DOES NOTHING FOR NOW Whether to use the same displacement for both x and y directions.

    Returns
    -------
    img: pyvips.Image

    """

    # TLDR: It's currently slow due to Box-Muller workaround, but will be solved in vips 8.16:
    # https://github.com/libvips/pyvips/issues/436
    # Image size: 24448 55293, camelyon16 normal 1
    # Mine: 340
    # Mine @ no box-muller: 170, 166 (flawed -> did not take into account gaus blurr)
    # Hans @ grid scale 8: 260
    # Hans @ grid scale 16: 257
    # Hans @ grid scale 64: 257

    width, height, channels = img.width, img.height, img.bands

    # Create a random displacement field, pyvips does not have uniform sampling (yet)
    # instead, use a Gaussian and convert using Box-Muller inverse
    z1 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)
    z2 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)

    # Compute box-muller inverse to get approximate uniform values
    dx = (-(z1 * z1 + z2 * z2) / 2).exp()
    dx = (2 * dx - 1).gaussblur(sigma) * alpha

    dy = (z1 / z2).atan()
    dy = (2 * dy - 1).gaussblur(sigma) * alpha

    grid = pyvips.Image.xyz(img.width, img.height)
    new_coords = grid + dx.bandjoin([dy])

    image = img.mapim(new_coords, interpolate=interpolation, background=background)

    return image


def pad_with_params(
    img: pyvips.Image,
    direction: str,
    width: int,
    height: int,
    border_mode: int = pyvips.enums.Extend.MIRROR,
    value: ImageColorType | None = None,
) -> pyvips.Image:
    return img.gravity(direction, width, height, extend=border_mode, background=value)


def bbox_vflip(bbox: BoxInternalType, rows: int, cols: int) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box vertically around the x-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, 1 - y_max, x_max, 1 - y_min


def bbox_hflip(bbox: BoxInternalType, rows: int, cols: int) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box horizontally around the y-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return 1 - x_max, y_min, 1 - x_min, y_max


def bbox_flip(bbox: BoxInternalType, d: int, rows: int, cols: int) -> BoxInternalType:
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        d: dimension. 0 for vertical flip, 1 for horizontal, -1 for transpose
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bbox


def bbox_transpose(
    bbox: KeypointInternalType, axis: int, rows: int, cols: int
) -> KeypointInternalType:  # skipcq: PYL-W0613
    """Transposes a bounding box along given axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        axis: 0 - main axis, 1 - secondary axis.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If axis not equal to 0 or 1.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    if axis not in {0, 1}:
        raise ValueError("Axis must be either 0 or 1.")
    if axis == 0:
        bbox = (y_min, x_min, y_max, x_max)
    if axis == 1:
        bbox = (1 - y_max, 1 - x_max, 1 - y_min, 1 - x_min)
    return bbox


@angle_2pi_range
def keypoint_vflip(keypoint: KeypointInternalType, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    angle = -angle
    return x, (rows - 1) - y, angle, scale


@angle_2pi_range
def keypoint_hflip(keypoint: KeypointInternalType, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint horizontally around the y-axis.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    angle = math.pi - angle
    return (cols - 1) - x, y, angle, scale


def keypoint_flip(keypoint: KeypointInternalType, d: int, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        d: Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        keypoint = keypoint_vflip(keypoint, rows, cols)
    elif d == 1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
    elif d == -1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
        keypoint = keypoint_vflip(keypoint, rows, cols)
    else:
        raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1")
    return keypoint


def keypoint_transpose(keypoint: KeypointInternalType) -> KeypointInternalType:
    """Rotate a keypoint by angle.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]

    if angle <= np.pi:
        angle = np.pi - angle
    else:
        angle = 3 * np.pi - angle

    return y, x, angle, scale


def _is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
    return np.allclose(matrix.params, np.eye(3, dtype=np.float32))


def warp_affine(
    image: pyvips.Image,
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: str,
    cval: int | float | Sequence[int] | Sequence[float],
    mode: str,
    output_shape: Sequence[int],
) -> pyvips.Image:
    if _is_identity_matrix(matrix):
        return image

    dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))

    image = image.affine(
        matrix.params[:2, :2].flatten().tolist(),
        odx=matrix.params[0, 2],
        ody=matrix.params[1, 2],
        oarea=(0, 0, dsize[0], dsize[1]),
        extend=mode,
        interpolate=interpolation,
        background=cval,
    )
    return image


def rotation2DMatrixToEulerAngles(matrix: np.ndarray, y_up: bool = False) -> float:
    """
    Args:
        matrix (np.ndarray): Rotation matrix
        y_up (bool): is Y axis looks up or down
    """
    if y_up:
        return np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.arctan2(-matrix[1, 0], matrix[0, 0])


@angle_2pi_range
def keypoint_affine(
    keypoint: KeypointInternalType,
    matrix: skimage.transform.ProjectiveTransform,
    scale: dict,
) -> KeypointInternalType:
    if _is_identity_matrix(matrix):
        return keypoint

    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix.params[:2]).squeeze()
    a += rotation2DMatrixToEulerAngles(matrix.params[:2])
    s *= np.max([scale["x"], scale["y"]])
    return x, y, a, s


def bbox_affine(
    bbox: BoxInternalType,
    matrix: skimage.transform.ProjectiveTransform,
    rotate_method: str,
    rows: int,
    cols: int,
    output_shape: Sequence[int],
) -> BoxInternalType:
    if _is_identity_matrix(matrix):
        return bbox
    x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
    if rotate_method == "largest_box":
        points = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
        )
    elif rotate_method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        data = np.arange(0, 360, dtype=np.float32)
        x = w * np.sin(np.radians(data)) + (w + x_min - 0.5)
        y = h * np.cos(np.radians(data)) + (h + y_min - 0.5)
        points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    else:
        raise ValueError(f"Method {rotate_method} is not a valid rotation method.")
    points = skimage.transform.matrix_transform(points, matrix.params)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return normalize_bbox((x_min, y_min, x_max, y_max), output_shape[0], output_shape[1])
