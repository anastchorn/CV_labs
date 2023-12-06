import cv2
import numpy as np
import os

def erosion(image, kernel, iterations=1):
    result = np.zeros_like(image)

    for _ in range(iterations):
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                match = True
                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        if kernel[m, n] == 1 and image[i - 1 + m, j - 1 + n] != 255:
                            match = False
                            break
                result[i, j] = 255 if match else 0

    return result

def dilation(image, kernel):
    result = np.zeros_like(image)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            dilate_value = False
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    if kernel[m, n] == 1 and image[i - 1 + m, j - 1 + n] == 255:
                        dilate_value = True
                        break
            result[i, j] = 255 if dilate_value else 0

    return result

def opening(image, kernel):
    eroded = erosion(image, kernel)
    result = dilation(eroded, kernel)

    return result

def closing(image, kernel):
    dilated = dilation(image, kernel)
    result = np.zeros_like(image, dtype=np.uint8)


    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sum_neighbors = np.sum(dilated[i - 1:i + 2, j - 1:j + 2] * kernel)
            result[i, j] = 255 if sum_neighbors > 0 else 0

    return result

def gradient_magnitude(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = convolve(image, kernel_x)
    gradient_y = convolve(image, kernel_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = (magnitude / np.max(magnitude)) * 255
    return magnitude.astype(np.uint8)

def convolve(image, kernel):
    return cv2.filter2D(image, cv2.CV_64F, kernel)

def find_edges(image):

    if len(image.shape) == 3:
        image = np.mean(image, axis=-1).astype(np.uint8)
    edges = gradient_magnitude(image)
    return edges


image_path = 'binary_image_2.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


output_directory = 'output_images'
os.makedirs(output_directory, exist_ok=True)


kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], np.uint8)

# apply morphological operations and edge detection
eroded_image = (erosion(original_image, kernel) > 0).astype(np.uint8) * 255
dilated_image = (dilation(original_image, kernel) > 0).astype(np.uint8) * 255
opened_image = (opening(original_image, kernel) > 0).astype(np.uint8) * 255
closed_image = (closing(original_image, kernel) > 0).astype(np.uint8) * 255
edges_image = find_edges(original_image)

# save the results in the output directory
cv2.imwrite(os.path.join(output_directory, 'binary_image1.jpg'), original_image)
cv2.imwrite(os.path.join(output_directory, 'eroded_image1.jpg'), eroded_image)
cv2.imwrite(os.path.join(output_directory, 'dilated_image1.jpg'), dilated_image)
cv2.imwrite(os.path.join(output_directory, 'opened_image1.jpg'), opened_image)
cv2.imwrite(os.path.join(output_directory, 'closed_image1.jpg'), closed_image)
cv2.imwrite(os.path.join(output_directory, 'edges_image1.jpg'), edges_image)

