import cv2
import numpy as np

# load an image
image = cv2.imread('my_image1.jpg', 0)  # read the image as grayscale (if needed)

# function to save an image
def save_image(output_image, filename):
    cv2.imwrite(filename, output_image)

# 1. shift the image 10 pixels to the right and 20 pixels down
def shift_image(image):
    height, width = image.shape
    shifted_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            # calculate new pixel coordinates
            new_x, new_y = x + 10, y + 20
            # check if new coordinates are within image bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                shifted_image[new_y, new_x] = image[y, x]
    return shifted_image

translated_image = shift_image(image)
save_image(translated_image, 'translated_image1.jpg')

# 2. color inversion
inverted_image = 255 - image
save_image(inverted_image, 'inverted_image1.jpg')

# 3. gaussian blur
def gaussian_blur(image, kernel_size):
    blurred_image = np.copy(image)
    k = kernel_size // 2
    height, width = image.shape
    for y in range(k, height - k):
        for x in range(k, width - k):
            # calculate the mean value of pixels in a window as a blurring effect
            blurred_image[y, x] = np.mean(image[y-k:y+k+1, x-k:x+k+1])
    return blurred_image

blurred_image = gaussian_blur(image, 11)
save_image(blurred_image, 'blurred_image1.jpg')

# 4. diagonal motion blur
def motion_blur(image, kernel_size):
    kernel = np.eye(kernel_size) / kernel_size
    motion_blur_image = cv2.filter2D(image, -1, kernel)
    return motion_blur_image

motion_blur_image = motion_blur(image, 7)
save_image(motion_blur_image, 'motion_blur_image1.jpg')

# 5. function to apply the "Sharpening" filter
def custom_sharpen_image(image):
    height, width = image.shape
    sharpened_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = 10 * image[y, x]
            neighbors = image[y-1:y+2, x-1:x+2].flatten()
            # calculate the difference between the central pixel and its neighbors
            sharpened_pixel = center - neighbors.sum()
            sharpened_image[y, x] = np.clip(sharpened_pixel, 0, 255)  # Use np.clip to constrain values
    return sharpened_image

sharpened_image = custom_sharpen_image(image)
save_image(sharpened_image, 'sharpened_image1.jpg')

# 6. function to apply the "Sobel Filter"
def custom_sobel_filter(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    height, width = image.shape
    sobel_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pixel_region = image[y-1:y+2, x-1:x+2]
            gradient_x = (sobel_x * pixel_region).sum()
            gradient_y = (sobel_y * pixel_region).sum()
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            sobel_image[y, x] = np.clip(gradient_magnitude, 0, 255)
    return sobel_image

sobel_image = custom_sobel_filter(image)
save_image(sobel_image, 'sobel_image1.jpg')

# 7. function to apply the "Edge Detection" filter
def custom_edge_detection(image, low_threshold, high_threshold):
    gradient_magnitude = custom_sobel_filter(image)
    height, width = image.shape
    edge_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if gradient_magnitude[y, x] >= high_threshold:
                edge_image[y, x] = 255
            elif gradient_magnitude[y, x] >= low_threshold:
                edge_image[y, x] = 50  # pixels meeting the low threshold are set to 50 (gray)

    return edge_image

edge_image = custom_edge_detection(image, 100, 200)
save_image(edge_image, 'edge_image1.jpg')

# 8. Function to apply my filter
def adjust_saturation(image, factor):
    # Iterate over each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract pixel value
            pixel_value = image[y, x]

            # Adjust saturation for the single channel
            adjusted_value = int(max(0, min(255, pixel_value * factor)))

            # Update the pixel with adjusted value
            image[y, x] = adjusted_value
    return

# Adjust saturation by a factor (e.g., 1.5 for increasing saturation)
adjust_saturation(image, 1.5)
# Save the result
save_image(image, 'saturated_image1.jpg')

