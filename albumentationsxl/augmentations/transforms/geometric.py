"""
adapted from
https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/geometric/transforms.py
The transformations are basically the same as albumentationsxl, replacing numpy operations with pyvips wherever needed

"""

import pyvips
import random

import numpy as np
import skimage.transform

from enum import Enum
from collections.abc import Sequence
from typing import Optional

from ..functional import geometric as F
from ..functional.misc import bbox_from_mask
from ...core.bbox_utils import denormalize_bbox, normalize_bbox
from ...core.transforms_interface import BoxInternalType, DualTransform, KeypointInternalType, ImageColorType, to_tuple

__all__ = ["VerticalFlip", "HorizontalFlip", "Flip", "Transpose", "ElasticTransform", "PadIfNeeded", "Affine"]


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, **params) -> pyvips.Image:
        return F.vflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_vflip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_vflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, **params) -> pyvips.Image:
        return F.hflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, d: int = 0, **params) -> pyvips.Image:
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return F.random_flip(img, d)

    def get_params(self):
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, **params) -> pyvips.Image:
        return F.transpose(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_transpose(keypoint)

    def get_transform_init_args_names(self):
        return ()


class ElasticTransform(DualTransform):
    """Elastic deformation of images as described in [Simard2003] .

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float): Magnitude/intensity of the displacement
        sigma (float): Gaussian filter parameter.
        interpolation (Pyvips Interpolate): Interpolation object that is used to specify the interpolation algorithm. Should be one of:
            pyvips.Interpolate.new(bilinear), pyvips.Interpolate.new(cubic), pyvips.Interpolate.new(nearest).
            Default: pyvips.Interpolate.new("bilinear").
        value (list of ints,
                list of floats): padding value for the background, applied to the image.
        mask_value (list of ints,
                    list of float): padding value for the background, applied to the mask.
        same_dxdy (boolean): DOES NOT WORK FOR NOW: Whether to use same random generated shift for x and y.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 50.0,
        interpolation: pyvips.Interpolate = pyvips.Interpolate.new("bilinear"),
        value: list = None,
        mask_value: list = None,
        always_apply: bool = False,
        same_dxdy: bool = False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        print("Elastic transform is only supported for image-only datasets, do not use with bounding boxes/masks!")
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.value = [255, 255, 255] if value is None else value
        self.mask_value = [0, 0, 0] if mask_value is None else mask_value
        self.same_dxdy = same_dxdy

    def apply(self, img, random_state=None, interpolation=pyvips.Interpolate.new("bilinear"), **params):
        return F.elastic_transform(img, self.alpha, self.sigma, interpolation, self.value, self.same_dxdy)

    def apply_to_mask(self, img, random_state=None, **params):
        return F.elastic_transform(
            img, self.alpha, self.sigma, pyvips.Interpolate.new("nearest"), self.mask_value, self.same_dxdy
        )

    def apply_to_bbox(self, bbox, random_state=None, **params):
        """Does not work for now: All numpy arrays so will crash when used on pyvips functions"""
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.elastic_transform(
            mask, self.alpha, self.sigma, pyvips.Interpolate.new("nearest"), self.mask_value, self.same_dxdy
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {"random_state": random.randint(0, 10000)}

    def get_transform_init_args_names(self):
        return ("alpha", "sigma", "interpolation", "value", "mask_value", "same_dxdy")


class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            Default: PositionType.CENTER.
        border_mode (pyvips enums extend): pyvips border mode.
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value for mask if border_mode is pyvips.enums.Extend.constant.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    class PositionType(Enum):
        CENTER = "centre"
        TOP_LEFT = "north-west"
        TOP_RIGHT = "north-east"
        BOTTOM_LEFT = "south-west"
        BOTTOM_RIGHT = "south-east"

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: PositionType | str = PositionType.CENTER,
        border_mode: int = pyvips.enums.Extend.BACKGROUND,
        value: ImageColorType | None = None,
        mask_value: ImageColorType | None = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right) = self.__update_position_params(
            h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        )

        new_width = cols + w_pad_left + w_pad_right
        new_height = rows + h_pad_top + h_pad_bottom

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
                "new_height": new_height,
                "new_width": new_width,
            }
        )
        return params

    def apply(self, img: pyvips.Image, new_width: int = 0, new_height: int = 0, **params) -> pyvips.Image:
        return F.pad_with_params(
            img,
            direction=self.position.value,
            width=new_width,
            height=new_height,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(self, img: pyvips.Image, new_width: int = 0, new_height: int = 0, **params) -> pyvips.Image:
        return F.pad_with_params(
            img,
            direction=self.position.value,
            width=new_width,
            height=new_height,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        rows: int = 0,
        cols: int = 0,
        **params,
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        **params,
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        return h_top, h_bottom, w_left, w_right


class Affine(DualTransform):
    """Augmentation to apply affine transformations to images.
    This is mostly a wrapper around the corresponding classes and functions in OpenCV.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a defined content, e.g.
    if the image is translated to the left, pixels are created on the right.
    A method has to be defined to deal with these pixel values.
    The parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameters `interpolation` and
    `mask_interpolation` deals with the method of interpolation used for this.

    Args:
        scale (number, tuple of number or dict): Scaling factor to use, where ``1.0`` denotes "no change" and
            ``0.5`` is zoomed out to ``50`` percent of the original size.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That the same range will be used for both x- and y-axis. To keep the aspect ratio, set
                  ``keep_ratio=True``, then the same value will be used for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes. Note that when
                  the ``keep_ratio=True``, the x- and y-axis ranges should be the same.
        translate_percent (None, number, tuple of number or dict): Translation as a fraction of the image height/width
            (x-translation, y-translation), where ``0`` denotes "no change"
            and ``0.5`` denotes "half of the axis size".
                * If ``None`` then equivalent to ``0.0`` unless `translate_px` has a value other than ``None``.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That sampled fraction value will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        translate_px (None, int, tuple of int or dict): Translation in pixels.
                * If ``None`` then equivalent to ``0`` unless `translate_percent` has a value other than ``None``.
                * If a single int, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from
                  the discrete interval ``[a..b]``. That number will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        rotate (number or tuple of number): Rotation in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``. Rotation happens around the *center* of the image,
            not the top left corner as in some other frameworks.
                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``
                  and used as the rotation value.
        shear (number, tuple of number or dict): Shear in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``, with reasonable values being in the range of ``[-45, 45]``.
                * If a number, then that value will be used for all images as
                  the shear on the x-axis (no shear on the y-axis will be done).
                * If a tuple ``(a, b)``, then two value will be uniformly sampled per image
                  from the interval ``[a, b]`` and be used as the x- and y-shear value.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        interpolation (pyvips.Interpolate): Pyvips interpolation GObject.
        mask_interpolation (pyvips.Interpolate): Pyvips interpolation GObject.
        cval (number or sequence of number): The constant value to use when filling in newly created pixels.
            (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
            on the left of the image).
            The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
        cval_mask (number or tuple of number): Same as cval but only for masks.
        mode (str): pyvips border flag/enum.
        fit_output (bool): If True, the image plane size and position will be adjusted to tightly capture
            the whole image after affine transformation (`translate_percent` and `translate_px` are ignored).
            Otherwise (``False``),  parts of the transformed image may end up outside the image plane.
            Fitting the output shape can be useful to avoid corners of the image being outside the image plane
            after applying rotations. Default: False
        keep_ratio (bool): When True, the original aspect ratio will be kept when the random scale is applied.
                           Default: False.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or
            "ellipse"[1].
            Default: "largest_box"
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Reference:
        [1] https://arxiv.org/abs/2109.13488
    """

    def __init__(
        self,
        scale: float | Sequence[float] | dict | None = None,
        translate_percent: float | Sequence[float] | dict | None = None,
        translate_px: int | Sequence[int] | dict | None = None,
        rotate: float | Sequence[float] | dict | None = None,
        shear: float | Sequence[float] | dict | None = None,
        interpolation: pyvips.Interpolate = pyvips.Interpolate.new("bilinear"),
        mask_interpolation: pyvips.Interpolate = pyvips.Interpolate.new("nearest"),
        cval: int | float | Sequence[int] | Sequence[float] = 255,
        cval_mask: int | float | Sequence[int] | Sequence[float] = 0,
        mode: str = pyvips.enums.Extend.BACKGROUND,
        fit_output: bool = False,
        keep_ratio: bool = False,
        rotate_method: str = "largest_box",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        params = [scale, translate_percent, translate_px, rotate, shear]
        if all([p is None for p in params]):
            scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)}
            translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
            rotate = (-15, 15)
            shear = {"x": (-10, 10), "y": (-10, 10)}
        else:
            scale = scale if scale is not None else 1.0
            rotate = rotate if rotate is not None else 0.0
            shear = shear if shear is not None else 0.0

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.scale = self._handle_dict_arg(scale, "scale")
        self.translate_percent, self.translate_px = self._handle_translate_arg(translate_px, translate_percent)
        self.rotate = to_tuple(rotate, rotate)
        self.fit_output = fit_output
        self.shear = self._handle_dict_arg(shear, "shear")
        self.keep_ratio = keep_ratio
        self.rotate_method = rotate_method

        if self.keep_ratio and self.scale["x"] != self.scale["y"]:
            raise ValueError(
                "When keep_ratio is True, the x and y scale range should be identical. got {}".format(self.scale)
            )

    def get_transform_init_args_names(self):
        return (
            "interpolation",
            "mask_interpolation",
            "cval",
            "mode",
            "scale",
            "translate_percent",
            "translate_px",
            "rotate",
            "fit_output",
            "shear",
            "cval_mask",
            "keep_ratio",
            "rotate_method",
        )

    @staticmethod
    def _handle_dict_arg(val: float | Sequence[float] | dict, name: str, default: float = 1.0):
        if isinstance(val, dict):
            if "x" not in val and "y" not in val:
                raise ValueError(
                    f'Expected {name} dictionary to contain at least key "x" or ' 'key "y". Found neither of them.'
                )
            x = val.get("x", default)
            y = val.get("y", default)
            return {"x": to_tuple(x, x), "y": to_tuple(y, y)}
        return {"x": to_tuple(val, val), "y": to_tuple(val, val)}

    @classmethod
    def _handle_translate_arg(
        cls,
        translate_px: Optional[float | Sequence[float] | dict],
        translate_percent: Optional[float | Sequence[float] | dict],
    ):
        if translate_percent is None and translate_px is None:
            translate_px = 0

        if translate_percent is not None and translate_px is not None:
            raise ValueError(
                "Expected either translate_percent or translate_px to be " "provided, " "but neither of them was."
            )

        if translate_percent is not None:
            # translate by percent
            return cls._handle_dict_arg(translate_percent, "translate_percent", default=0.0), translate_px

        if translate_px is None:
            raise ValueError("translate_px is None.")
        # translate by pixels
        return translate_percent, cls._handle_dict_arg(translate_px, "translate_px")

    def apply(
        self,
        img: pyvips.Image,
        matrix: skimage.transform.ProjectiveTransform = None,
        output_shape: Sequence[int] = (),
        **params,
    ) -> pyvips.Image:
        return F.warp_affine(
            img,
            matrix,
            interpolation=self.interpolation,
            cval=self.cval,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_mask(
        self,
        img: pyvips.Image,
        matrix: skimage.transform.ProjectiveTransform = None,
        output_shape: Sequence[int] = (),
        **params,
    ) -> pyvips.Image:
        return F.warp_affine(
            img,
            matrix,
            interpolation=self.mask_interpolation,
            cval=self.cval_mask,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        matrix: skimage.transform.ProjectiveTransform = None,
        rows: int = 0,
        cols: int = 0,
        output_shape: Sequence[int] = (),
        **params,
    ) -> BoxInternalType:
        return F.bbox_affine(bbox, matrix, self.rotate_method, rows, cols, output_shape)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        matrix: skimage.transform.ProjectiveTransform | None = None,
        scale: dict | None = None,
        **params,
    ) -> KeypointInternalType:
        assert scale is not None and matrix is not None
        return F.keypoint_affine(keypoint, matrix=matrix, scale=scale)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        h, w = params["image"].height, params["image"].width

        translate: dict[str, int | float]
        if self.translate_px is not None:
            translate = {key: random.randint(*value) for key, value in self.translate_px.items()}
        elif self.translate_percent is not None:
            translate = {key: random.uniform(*value) for key, value in self.translate_percent.items()}
            translate["x"] = translate["x"] * w
            translate["y"] = translate["y"] * h
        else:
            translate = {"x": 0, "y": 0}

        # Look to issue https://github.com/albumentations-team/albumentations/issues/1079
        shear = {key: -random.uniform(*value) for key, value in self.shear.items()}
        scale = {key: random.uniform(*value) for key, value in self.scale.items()}
        if self.keep_ratio:
            scale["y"] = scale["x"]

        # Look to issue https://github.com/albumentations-team/albumentations/issues/1079
        rotate = -random.uniform(*self.rotate)

        # for images we use additional shifts of (0.5, 0.5) as otherwise
        # we get an ugly black border for 90deg rotations
        shift_x = w / 2 - 0.5
        shift_y = h / 2 - 0.5

        matrix_to_topleft = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
        matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
        matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
        matrix_transforms = skimage.transform.AffineTransform(
            scale=(scale["x"], scale["y"]),
            translation=(translate["x"], translate["y"]),
            rotation=np.deg2rad(rotate),
            shear=np.deg2rad(shear["x"]),
        )
        matrix_to_center = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (
            matrix_to_topleft
            + matrix_shear_y_rot
            + matrix_shear_y
            + matrix_shear_y_rot_inv
            + matrix_transforms
            + matrix_to_center
        )
        if self.fit_output:
            matrix, output_shape = self._compute_affine_warp_output_shape(
                matrix, (params["image"].height, params["image"].width, params["image"].bands)
            )
        else:
            output_shape = (params["image"].height, params["image"].width, params["image"].bands)

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "output_shape": output_shape,
        }

    @staticmethod
    def _compute_affine_warp_output_shape(
        matrix: skimage.transform.ProjectiveTransform, input_shape: Sequence[int]
    ) -> tuple[skimage.transform.ProjectiveTransform, Sequence[int]]:
        height, width = input_shape[:2]

        if height == 0 or width == 0:
            return matrix, input_shape

        # determine shape of output image
        corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        corners = matrix(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_height = maxr - minr + 1
        out_width = maxc - minc + 1
        if len(input_shape) == 3:
            output_shape = np.ceil((out_height, out_width, input_shape[2]))
        else:
            output_shape = np.ceil((out_height, out_width))
        output_shape_tuple = tuple([int(v) for v in output_shape.tolist()])
        # fit output image in new shape
        translation = (-minc, -minr)
        matrix_to_fit = skimage.transform.SimilarityTransform(translation=translation)
        matrix = matrix + matrix_to_fit
        return matrix, output_shape_tuple
