import pyvips
from collections.abc import Iterable


def cutout(img: pyvips.Image, holes: Iterable[tuple[int, int, int, int]], fill_value: int | float = 0) -> pyvips.Image:
    for x1, y1, x2, y2 in holes:
        img = img.draw_rect([fill_value, fill_value, fill_value], x1, y1, x2 - x1, y2 - y1, fill=True)

    return img


def channel_dropout(
    img: pyvips.Image, channels_to_drop: int | tuple[int, ...] | pyvips.Image, fill_value: int | float = 0
) -> pyvips.Image:
    if img.bands == 2 or img.bands == 1:
        raise NotImplementedError("Only one channel. ChannelDropout is not defined.")

    format = img.format
    a = [1, 1, 1] # multiplication for linear
    b = [0, 0, 0] # addition

    channels_to_drop = list(channels_to_drop)
    for channel in channels_to_drop:
        a[channel] = 0
        b[channel] = fill_value

    print(a,b)
    return img.linear(a,b, uchar=(format == "uchar"))