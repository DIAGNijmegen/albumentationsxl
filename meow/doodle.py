import pyvips
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


mask = np.ones((30, 30), dtype="uint8")
mask = pyvips.Image.new_from_array(mask.tolist(), scale=30 * 30)


image = np.array(Image.open("../images/coco_cat_dog.jpg"))

plt.imshow(image)
plt.show()


pyvips_image = pyvips.Image.new_from_array(image)
print("pyvips image dimensions before resizing", pyvips_image.width, pyvips_image.height)
height = pyvips_image.height
width = pyvips_image.width
hscale= 1.5
vscale = 0.8
pyvips_image = pyvips_image.resize(hscale, vscale=vscale, kernel=pyvips.Kernel.LINEAR)

print("pyvips image dimensions after resizing", pyvips_image.width, pyvips_image.height)
print("pyvips image dimensions proportions after resizing", pyvips_image.width / width, pyvips_image.height / height)

plt.imshow(pyvips_image.numpy())
plt.show()
