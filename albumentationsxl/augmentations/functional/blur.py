import pyvips
import numpy as np

__all__ = ["blur", "gaussian_blur", "median_blur"]

# TODO: harmonize blurring methods that use pyvips convolutions towards common inputs, perhaps a common function?


def blur(img: pyvips.Image, ksize: int) -> pyvips.Image:
    """

    Parameters
    ----------
    img : pyvips.Image
        The image to be blurred, can be float or np.uint8
    ksize: int
        the kernel size of the box filter

    Returns
    -------
    img: pyvips.Image
        The blurred image according to the box filter with kernel size ksize
    """
    mask = np.ones((ksize, ksize), dtype="uint8")
    mask = pyvips.Image.new_from_list(mask.tolist(), scale=ksize * ksize)
    return img.conv(mask, precision="integer")


def gaussian_blur(img: pyvips.Image, sigma: float, min_amplitude: float) -> pyvips.Image:
    return img.gaussblur(sigma, min_ampl=min_amplitude)


def median_blur(img: pyvips.Image, ksize: int) -> pyvips.Image:
    return img.median(ksize)



