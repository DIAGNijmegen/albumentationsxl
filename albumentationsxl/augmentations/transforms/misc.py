import pyvips
import random
import warnings
import numpy as np

from types import LambdaType

from .. import functional as F
from ...core.utils import format_args
from ...core.transforms_interface import (
    ImageOnlyTransform,
    NoOp,
    to_tuple,
)

__all__ = ["Sharpen", "Emboss", "Lambda"]


class Sharpen(ImageOnlyTransform):
    """Sharpen the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness ((float, float)): range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self, alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5
    ):
        super(Sharpen, self).__init__(always_apply, p)
        self.alpha = self.__check_values(
            to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0)
        )
        self.lightness = self.__check_values(to_tuple(lightness, 0.0), name="lightness")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_sharpening_matrix(alpha_sample, lightness_sample):
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness_sample, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self):
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(
            alpha_sample=alpha, lightness_sample=lightness
        )
        return {
            "sharpening_matrix": pyvips.Image.new_from_list(sharpening_matrix.tolist())
        }

    def apply(
        self, img: pyvips.Image, sharpening_matrix: pyvips.Image | None = None, **params
    ):
        # Can we do integer convolutions here?
        img = img.cast("float") / 255
        img = img.conv(sharpening_matrix)
        return (img * 255).cast("uchar")

    def get_transform_init_args_names(self):
        return "alpha", "lightness"


class Emboss(ImageOnlyTransform):
    """Emboss the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the embossed image. At 0, only the original image is
            visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength ((float, float)): strength range of the embossing. Default: (0.2, 0.7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self, alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5
    ):
        super(Emboss, self).__init__(always_apply, p)
        self.alpha = self.__check_values(
            to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0)
        )
        self.strength = self.__check_values(to_tuple(strength, 0.0), name="strength")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_emboss_matrix(alpha_sample, strength_sample):
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [
                [-1 - strength_sample, 0 - strength_sample, 0],
                [0 - strength_sample, 1, 0 + strength_sample],
                [0, 0 + strength_sample, 1 + strength_sample],
            ],
            dtype=np.float32,
        )
        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self):
        alpha = random.uniform(*self.alpha)
        strength = random.uniform(*self.strength)
        emboss_matrix = self.__generate_emboss_matrix(
            alpha_sample=alpha, strength_sample=strength
        )
        print(emboss_matrix)
        return {"emboss_matrix": pyvips.Image.new_from_list(emboss_matrix.tolist())}

    def apply(self, img, emboss_matrix=None, **params):
        img = img.cast("float") / 255
        img = img.conv(emboss_matrix)
        return (img * 255).cast("uchar")

    def get_transform_init_args_names(self):
        return "alpha", "strength"


class Lambda(NoOp):
    """A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optinal arguments like interpolation method, image size, etc:

    Args:
        image (callable): Image transformation function.
        mask (callable): Mask transformation function.
        keypoint (callable): Keypoint transformation function.
        bbox (callable): BBox transformation function.
        always_apply (bool): Indicates whether this transformation should be always applied.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        Any
    """

    def __init__(
        self,
        image=None,
        mask=None,
        keypoint=None,
        bbox=None,
        name=None,
        always_apply=False,
        p=1.0,
    ):
        super(Lambda, self).__init__(always_apply, p)

        self.name = name
        self.custom_apply_fns = {
            target_name: F.noop for target_name in ("image", "mask", "keypoint", "bbox")
        }
        for target_name, custom_apply_fn in {
            "image": image,
            "mask": mask,
            "keypoint": keypoint,
            "bbox": bbox,
        }.items():
            if custom_apply_fn is not None:
                if (
                    isinstance(custom_apply_fn, LambdaType)
                    and custom_apply_fn.__name__ == "<lambda>"
                ):
                    warnings.warn(
                        "Using lambda is incompatible with multiprocessing. "
                        "Consider using regular functions or partial()."
                    )

                self.custom_apply_fns[target_name] = custom_apply_fn

    def apply(self, img, **params):
        fn = self.custom_apply_fns["image"]
        return fn(img, **params)

    def apply_to_mask(self, mask, **params):
        fn = self.custom_apply_fns["mask"]
        return fn(mask, **params)

    def apply_to_bbox(self, bbox, **params):
        fn = self.custom_apply_fns["bbox"]
        return fn(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        fn = self.custom_apply_fns["keypoint"]
        return fn(keypoint, **params)

    @classmethod
    def is_serializable(cls):
        return False

    def _to_dict(self):
        if self.name is None:
            raise ValueError(
                "To make a Lambda transform serializable you should provide the `name` argument, "
                "e.g. `Lambda(name='my_transform', image=<some func>, ...)`."
            )
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}

    def __repr__(self):
        state = {"name": self.name}
        state.update(self.custom_apply_fns.items())
        state.update(self.get_base_init_args())
        return "{name}({args})".format(
            name=self.__class__.__name__, args=format_args(state)
        )
