import numpy as np
from PIL import Image


width, height = 200, 100
image_array = np.zeros((height, width, 3), dtype=np.uint8)


image_array[30:70, 50:150] = [255, 0, 0]


image = Image.fromarray(image_array)


image.save("simple_image.png")


image.show()
