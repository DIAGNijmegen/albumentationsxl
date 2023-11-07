import random
import pyvips

from typing import Any
import cv2
import numpy as np
from ..functional.blur import blur, gaussian_blur, motion_blur

from albumentationsxl.core.transforms_interface import (
    ImageOnlyTransform,
    ScaleFloatType,
    ScaleIntType,
    to_tuple,
)


__all__ = ["Blur", "MotionBlur", "GaussianBlur"]


class Blur(ImageOnlyTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit: ScaleIntType = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, img: pyvips.Image, ksize: int = 3, **params) -> pyvips.Image:
        return blur(img, ksize)

    def get_params(self) -> dict[str, Any]:
        return {"ksize": int(random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2))))}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("blur_limit",)


class MotionBlur(Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        allow_shifted: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.allow_shifted = allow_shifted

        if not allow_shifted and self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return super().get_transform_init_args_names() + ("allow_shifted",)

    def apply(self, img: pyvips.Image, kernel: pyvips.Image = None, **params) -> pyvips.Image:  # type: ignore
        return img.conv(kernel, precision="integer")

    def get_params(self) -> dict[str, Any]:
        ksize = random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if x1 == x2:
            y1, y2 = random.sample(range(ksize), 2)
        else:
            y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)

        def make_odd_val(v1, v2):
            len_v = abs(v1 - v2) + 1
            if len_v % 2 != 1:
                if v2 > v1:
                    v2 -= 1
                else:
                    v1 -= 1
            return v1, v2

        if not self.allow_shifted:
            x1, x2 = make_odd_val(x1, x2)
            y1, y2 = make_odd_val(y1, y2)

            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            center = ksize / 2 - 0.5
            dx = xc - center
            dy = yc - center
            x1, x2 = [int(i - dx) for i in [x1, x2]]
            y1, y2 = [int(i - dy) for i in [y1, y2]]

        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

        print("kernel during aug", kernel)
        #kernel = kernel.astype(np.float32) / np.sum(kernel)
        kernel = kernel.astype(np.uint8)
        kernel = pyvips.Image.new_from_list(kernel.tolist(), scale=np.sum(kernel))
        # Normalize kernel
        return {"kernel": kernel}


class GaussianBlur(ImageOnlyTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    This implementation is not faithful to albumentation's gaussian blur. CV2 libraries have a predefined kernel
    or sigma, or calculate either the radius/sigma from the given kernel size. Pyvips defines an amplitude that
    determines the accuracy of the mask: Given a certain accuracy, the kernel size may change given varying levels
    of sigma

    In short: opencv sets the speed of the algorithm with a predefined kernel size, and the accuracy of the mask
    differs with sigma. Pyvips sets the accuracy, and the kernel size will change with varying sigma.

    Reference: https://github.com/libvips/libvips/discussions/3038

    Args:
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        min_amplitude (float): must be in range [0,inf)
            accuracy of the mask, implicitly determines radius/kernel size. Lower values typically lead to higher accuracy
            and larger kernels, at the cost of speed. Recommended to keep this at default of 0.2
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        sigma_limit: ScaleFloatType = 0,
        min_amplitude: float = 0.2,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)
        self.min_amplitude = min_amplitude

    def apply(self, img: pyvips.Image, min_amplitude: float = 0.2, sigma: float = 0, **params) -> pyvips.Image:
        return gaussian_blur(img, sigma=sigma, min_amplitude=self.min_amplitude)

    def get_params(self) -> dict[str, float]:
        return {"sigma": random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "sigma_limit", "min_amplitude"
