import numpy as np
import random
import cv2


def sp_noise_color(image, prob=0.03, white=[255,255,255], black=[0,0,0]):
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i,j,:] = black
            elif rdn > thres:
                image[i,j,:] = white
    return image



def box_average_filter(image, ksize=3):
    pad_size = ksize // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, :]

            for k in range(3):
                filtered_image[i - pad_size, j - pad_size, k] = np.mean(window[:, :, k])

    return filtered_image.astype(np.uint8)



def median_filter(image, ksize=3):
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





original_image = cv2.imread('my_image1.jpg')
intense_noise = sp_noise_color(original_image, 0.07)



int_med_image = median_filter(intense_noise)
int_weighted_median = weighted_median_filter(intense_noise)
int_box_image = box_average_filter(intense_noise)

cv2.imshow('Noisy Image', intense_noise)
cv2.imshow('Box Image', int_box_image)
cv2.imshow('Mediana Image',int_med_image)
cv2.imshow('Filtered Image (Weighted Median)', int_weighted_median)

cv2.waitKey(0)
cv2.destroyAllWindows()


