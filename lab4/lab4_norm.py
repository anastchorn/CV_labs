import numpy as np
import cv2

def norm_noise_color(image, mean=0, var=0.1, a=0.5):
    '''
    Add gaussian noise to color image

    image: Numpy 2D array
    mean: scalar - mean
    var: scalar - variance
    a: scalar [0-1] - alpha blend
    returns: Numpy 2D array
    '''
    sigma = var**0.5

    row,col,ch= image.shape[:3]
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = a*image + (1-a)*gauss

    noisy = noisy-np.min(noisy)
    noisy = 255*(noisy/np.max(noisy))
    return noisy.astype(np.uint8)


def box_average_filter(image, window_size):
    pad_size = window_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, :]

            for k in range(3):
                filtered_image[i - pad_size, j - pad_size, k] = np.mean(window[:, :, k])

    return filtered_image.astype(np.uint8)



def median_filter(image, ksize):
    pad_size = ksize // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, :]

            for k in range(3):
                filtered_image[i - pad_size, j - pad_size, k] = np.median(window[:, :, k])

    return filtered_image.astype(np.uint8)


def weighted_median_filter(image, ksize=3):
    weights = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])
    scaling_factor = 10
    weights = (weights / weights.sum() * scaling_factor).astype(int)


    pad_size = ksize // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'reflect')

    new_image = np.zeros_like(image)


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            roi = padded_image[i:i+ksize, j:j+ksize]

            for k in range(3):
                weighted_values = np.repeat(roi[:, :, k].flatten(), weights.flatten())
                weighted_median = np.median(weighted_values)
                new_image[i, j, k] = weighted_median

    return new_image.astype(image.dtype)




original_image = cv2.imread('alley_image.jpg')
norm_noise=norm_noise_color(original_image, mean=0, var=20, a=0.1)



norm_med_image = median_filter(norm_noise, ksize=3)
norm_weighted_median = weighted_median_filter(norm_noise)
norm_box_image = box_average_filter(norm_noise, window_size=3)

cv2.imshow('Noisy Image', norm_noise)
cv2.imshow('Box Image', norm_box_image)
cv2.imshow('Mediana Image',norm_med_image)
cv2.imshow('Filtered Image (Weighted Median)', norm_weighted_median)

cv2.waitKey(0)
cv2.destroyAllWindows()
