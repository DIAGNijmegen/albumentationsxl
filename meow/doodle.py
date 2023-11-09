import pyvips
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
import cv2
from collections.abc import Sequence
from typing import Union
from albumentations.augmentations.utils import (
    _maybe_process_in_chunks,
    angle_2pi_range,
    clipped,
    preserve_channel_dim,
    preserve_shape,
)

matrix = skimage.transform.SimilarityTransform(
    [[0.98303776, 0.04138573, -8.36696622], [-0.02377169, 0.96355743, 5.5219933], [0.0, 0.0, 1.0]]
)


def _is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
    return np.allclose(matrix.params, np.eye(3, dtype=np.float32))


def warp_affine(
    image: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: int,
    cval: Union[int, float, Sequence[int], Sequence[float]],
    mode: int,
    output_shape: Sequence[int],
) -> np.ndarray:
    if _is_identity_matrix(matrix):
        return image

    dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix.params[:2], dsize=dsize, flags=interpolation, borderMode=mode, borderValue=cval
    )
    tmp = warp_fn(image)
    return tmp


# Transform the normal image
image = np.array(Image.open("../images/coco_cat_dog.jpg"))

plt.imshow(image)
plt.show()

new_image = warp_affine(
    image,
    matrix,
    interpolation=cv2.INTER_LINEAR,
    mode=cv2.BORDER_CONSTANT,
    cval=0,
    output_shape=[image.shape[0], image.shape[1]],
)
plt.imshow(new_image)
plt.show()

### Transform the pyvips image


pyvips_image = pyvips.Image.new_from_array(image)
pyvips_image = pyvips_image.affine(
    matrix.params[:2, :2].flatten().tolist(),
    interpolate=pyvips.Interpolate.new("nearest"),
    odx=matrix.params[0, 2],
    ody=matrix.params[1, 2],
    oarea=(0, 0, image.shape[1], image.shape[0]),
    background=255,
)

plt.imshow(pyvips_image.numpy())
plt.show()
