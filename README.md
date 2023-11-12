# AlbumentationsXL

AlbumentationsXL is a Python package for image augmentations fork of the [Albumentations](https://github.com/albumentations-team/albumentations/tree/master) package, altered to accomodate large images using the [libvips](https://github.com/libvips/libvips) and [pyvips](https://github.com/libvips/pyvips) libraries.

## Why AlbumentationsXL
Although albumentations covers a wide range of augmentations in a user-friendly format, it is not well-suited for large image inputs.
Since image operations are performed using Numpy, OpenCV and/or Pillow, images are loaded directly into memory, and this can cause out of memory errors on large inputs and transformations.

To solve this, libvips and its Python wrapper Pyvips can be used as the image handler, which is designed to process large images with low memory footprint, while remaining fast in execution.
In doing so, the AlbumentationsXL package has the ease-of-use of Albumentations, while being capable to handle inputs of any size. Several fields that often deal with large images are:

- Digital pathology: 
- Remote sensing (satelite imagery)


images in this domain are often .tif,.mrxs,.svs, etc. Pyvips can deal with most of these extensions (reading) without any problem, as long as the proper libvips dependencies are used.


## Table of contents
- [Installation](#installation)
- [Documentation](#documentation)
- [A simple example](#a-simple-example)
- [List of augmentations](#list-of-augmentations)
  - [Pixel-level transforms](#pixel-level-transforms)
  - [Spatial-level transforms](#spatial-level-transforms)



## Installation
Albumentations requires Python 3.10 or higher. To install the latest version from PyPI:

```
pip install -U albumentationsxl
```

Additionally, libvips 8.14.5 or higher, along with its Python bindings, pyvips 2.2.1 or higher, are needed as it is the core image handler. 
More information on how to build and install libvips/pyvips can be found on their respective documentation.

## Documentation
The full documentation is available at **[https://albumentations.ai/docs/](https://albumentations.ai/docs/)**.

## A simple example

```python
import albumentationsxl as A
import pyvips

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = pyvips.Image.new_from_file("image.jpg")

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"].numpy() # pyvips pipelines are not executed until they reach a target pipeline
```

## List of augmentations

### Pixel-level transforms
Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:

- [AdvancedBlur](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.AdvancedBlur)
- [Blur](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Blur)
- [ChannelDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/channel_dropout/#albumentations.augmentations.dropout.channel_dropout.ChannelDropout)
- [ColorJitter](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter)
- [Emboss](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Emboss)
- [FromFloat](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FromFloat)
- [GaussNoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise)
- [GaussianBlur](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.GaussianBlur)
- [HueSaturationValue](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue)
- [MedianBlur](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.MedianBlur)
- [MotionBlur](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.MotionBlur)
- [Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)
- [RandomBrightnessContrast](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)
- [RandomGamma](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma)
- [Sharpen](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Sharpen)

### Spatial-level transforms
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

| Transform                                                                                                                                                                       | Image | Masks | BBoxes | Keypoints |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: | :-------: |
| [Affine](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Affine)                             | ✓     | ✓     | ✓      | ✓         |
| [CenterCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CenterCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [CoarseDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#albumentations.augmentations.dropout.coarse_dropout.CoarseDropout)           | ✓     | ✓     |        | ✓         |
| [Crop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.Crop)                                         | ✓     | ✓     | ✓      | ✓         |
| [ElasticTransform](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ElasticTransform)         | ✓     | ✓     | ✓      |           |
| [Flip](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Flip)                                 | ✓     | ✓     | ✓      | ✓         |
| [GridDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/grid_dropout/#albumentations.augmentations.dropout.grid_dropout.GridDropout)                   | ✓     | ✓     |        |           |
| [HorizontalFlip](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.HorizontalFlip)             | ✓     | ✓     | ✓      | ✓         |
| [Lambda](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Lambda)                                                 | ✓     | ✓     | ✓      | ✓         |
| [LongestMaxSize](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.LongestMaxSize)                     | ✓     | ✓     | ✓      | ✓         |
| [NoOp](https://albumentations.ai/docs/api_reference/core/transforms_interface/#albumentations.core.transforms_interface.NoOp)                                                   | ✓     | ✓     | ✓      | ✓         |
| [PadIfNeeded](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded)                   | ✓     | ✓     | ✓      | ✓         |
| [RandomCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [RandomRotate90](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.RandomRotate90)                     | ✓     | ✓     | ✓      | ✓         |
| [RandomScale](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.RandomScale)                           | ✓     | ✓     | ✓      | ✓         |
| [Resize](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.Resize)                                     | ✓     | ✓     | ✓      | ✓         |
| [Rotate](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate)                                     | ✓     | ✓     | ✓      | ✓         |
| [SmallestMaxSize](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.SmallestMaxSize)                   | ✓     | ✓     | ✓      | ✓         |
| [Transpose](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Transpose)                       | ✓     | ✓     | ✓      | ✓         |
| [VerticalFlip](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.VerticalFlip)                 | ✓     | ✓     | ✓      | ✓         |