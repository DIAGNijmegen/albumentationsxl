import numpy as np
from typing import Callable
from typing_extensions import Concatenate, ParamSpec

from functools import wraps

from albumentationsxl.core.keypoints_utils import angle_to_2pi_range
from albumentationsxl.core.transforms_interface import KeypointInternalType


__all__ = [
    "MAX_VALUES_BY_DTYPE",
    "angle_2pi_range",
]

# Pyvips to numpy band conversion dictionary
format_to_dtype = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}

P = ParamSpec("P")

MAX_VALUES_BY_DTYPE = {"uchar": 255, "ushort": 65535, "uint": 65535, "float": 1.0, "double": 1.0}


def angle_2pi_range(
    func: Callable[Concatenate[KeypointInternalType, P], KeypointInternalType]
) -> Callable[Concatenate[KeypointInternalType, P], KeypointInternalType]:
    @wraps(func)
    def wrapped_function(keypoint: KeypointInternalType, *args: P.args, **kwargs: P.kwargs) -> KeypointInternalType:
        (x, y, a, s) = func(keypoint, *args, **kwargs)[:4]
        return (x, y, angle_to_2pi_range(a), s)

    return wrapped_function