import pyvips
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


mask = np.ones((30,30), dtype='uint8')
mask = pyvips.Image.new_from_array(mask.tolist(), scale=30*30)


image = np.array(Image.open("../images/coco_cat_dog.jpg"))

plt.imshow(image)
plt.show()


pyvips_image = pyvips.Image.new_from_array(image)
pyvips_image = pyvips_image.cast("ushort")

pyvips_image = pyvips_image * 255
print(pyvips_image.format)


lut = pyvips.Image.identity(bands=pyvips_image.bands, ushort=(pyvips_image.format == "ushort"))
#print(lut.numpy())

lut = (lut / 256).floor().cast("ushort")

#print(lut.numpy())

#print(pyvips_image.numpy())
pyvips_image = pyvips_image.maplut(lut)

#print(pyvips_image.numpy())
