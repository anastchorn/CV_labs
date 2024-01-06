import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convolution(image, kernel):

    image_height, image_width, channels = image.shape
    kernel_height, kernel_width = kernel.shape


    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1


    output_image = np.zeros((output_height, output_width, channels))


    for c in range(channels):
        for i in range(output_height):
            for j in range(output_width):
                output_image[i, j, c] = np.sum(image[i:i+kernel_height, j:j+kernel_width, c] * kernel)

    return output_image


def shift_image(image, shift_x, shift_y):
    shifted_image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
    return shifted_image



def invert_image(image):
    kernel=np.array([[0, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]])
    return convolution(image, kernel)

def gaussian_blur(image, size):

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*(size**2))) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*size**2)),
        (size, size)
    )
    kernel /= np.sum(kernel)

    return convolution(image, kernel)

def diagonal_motion_blur(image, size):
    kernel = np.eye(size) / size
    return convolution(image, kernel)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return convolution(image, kernel)

def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    gradient_x = convolution(image, kernel_x)
    gradient_y = convolution(image, kernel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_magnitude

def edge_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    return convolution(image, kernel)

def sepia_filter(image):
    # Матриця перетворення кольорів для сепії
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])

    # Застосування матриці перетворення
    sepia_image = np.dot(image, sepia_matrix.T)

    # Обмеження значень до діапазону [0, 255]
    sepia_image = np.clip(sepia_image, 0, 255)

    return sepia_image.astype(np.uint8)


image_path = 'cat_image.jpg'
original_image = np.array(Image.open(image_path))


shifted_image = shift_image(original_image, shift_x=10, shift_y=20)
inverted_image = invert_image(original_image)
blurred_image = gaussian_blur(original_image, size=11)
motion_blur_image = diagonal_motion_blur(original_image, size=7)
sharpened_image = sharpen_image(original_image)
sobel_filtered_image = sobel_filter(original_image)
edge_filtered_image = edge_filter(original_image)
sepia_filtered_image = sepia_filter(original_image)





fig, axes = plt.subplots(2, 4, figsize=(15, 8))

axes[0, 0].imshow(sepia_filtered_image.astype('uint8'))
axes[0, 0].set_title('Myfilter Image')

axes[0, 1].imshow(shifted_image.astype('uint8'))
axes[0, 1].set_title('Shifted Image')

axes[0, 2].imshow(inverted_image.astype('uint8'))
axes[0, 2].set_title('Inverted Image')

axes[0, 3].imshow(blurred_image.astype('uint8'))
axes[0, 3].set_title('Gaussian Blur')

axes[1, 0].imshow(motion_blur_image.astype('uint8'))
axes[1, 0].set_title('Motion Blur')

axes[1, 1].imshow(sharpened_image.astype('uint8'))
axes[1, 1].set_title('Sharpened Image')

axes[1, 2].imshow(sobel_filtered_image.astype('uint8'), cmap='gray')
axes[1, 2].set_title('Sobel Filter')

axes[1, 3].imshow(edge_filtered_image.astype('uint8'))
axes[1, 3].set_title('Edge Filter')

plt.tight_layout()
plt.show()
