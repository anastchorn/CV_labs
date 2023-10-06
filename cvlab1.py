import cv2
import numpy as np
# Конвертуємо кольорово зоображення в чб
def convert_to_bw(image):
    bw_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_color = image[i, j]
            gray_value = int(0.36 * pixel_color[2] + 0.53 * pixel_color[1] + 0.11 * pixel_color[0])
            bw_image[i, j] = gray_value
    return bw_image

# Бінарізуємо зоображеня
def binarize(image, threshold=150):
    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            if pixel_value < threshold:
                binary_mask[i, j] = 255
            else:
                binary_mask[i, j] = 0
    return binary_mask

# Вирізаємо об'єкт з кольорового зображення за допомогою маски
def cutout_object(original_image, binary_mask):
    cutout_image = np.zeros_like(original_image)
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            if binary_mask[i, j] == 255:
                cutout_image[i, j] = original_image[i, j]
    return cutout_image

# Перетворюємо зоображення
def process_image(my_input_image, border_value):
    input_image = cv2.imread(my_input_image)
    bw_image = convert_to_bw(input_image)
    binary_mask = binarize(bw_image, border_value)
    cutout_image = cutout_object(input_image, binary_mask)

    # Заміняємо фон на чорний
    cutout_image[np.where((cutout_image == [0, 0, 0]).all(axis=2))] = [0, 0, 0]

    # ОТримуємо номер фото з його назви
    image_number = my_input_image.split('_')[-1].split('.')[0]

    #Зберігаємо готові зоображеня з іхніми номерами
    cv2.imwrite( "bw_" + image_number + '.jpg', bw_image)
    cv2.imwrite("binary_image_" + image_number + '.jpg', binary_mask)
    cv2.imwrite( "cutoutimage_" + image_number + '.jpg', cutout_image)


my_input_image = 'ph_1.jpg'
border_value = 230


process_image(my_input_image, border_value)

