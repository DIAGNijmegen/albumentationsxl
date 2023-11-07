import pyvips
import numpy as np

__all__ = ["blur", "gaussian_blur", "motion_blur"]

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



def motion_blur2(img: pyvips.Image, kernel: pyvips.Image) -> pyvips.Image:
    # For large images, perform integer convolutions as much as possible
    img = img.cast("ushort")
    kernel = (kernel * 255).cast("uchar")

    # Instead of using math, which converts to float, build a lut in ushort format
    img = img.conv(kernel, precision="integer")

    # The lut is an 65536 * 3 array
    lut = pyvips.Image.identity(bands=img.bands, ushort=(img.format == "ushort"))
    lut = (lut / 256).floor().cast("ushort")

    # Map and bandjoin the image separately, or it will become a histogram interpretation
    img = img.maplut(lut)

    print(img.numpy())
    print(img)
    return img.cast("uchar")

def motion_blur(img: pyvips.Image, kernel: pyvips.Image) -> pyvips.Image:
    # For large images, perform integer convolutions as much as possible
    img = img.cast("ushort")
    kernel = (kernel * 255).cast("uchar")

    # Instead of using math, which converts to float, build a lut in ushort format
    img = img.conv(kernel, precision="integer")

    # The lut is an 65536 * 3 array
    lut = pyvips.Image.identity(bands=img.bands, ushort=(img.format == "ushort"))
    lut = (lut / 256).floor().cast("ushort")

    # Map and bandjoin the image separately, or it will become a histogram interpretation
    img = img.maplut(lut)

    print(img.numpy())
    print(img)
    return img.cast("uchar")