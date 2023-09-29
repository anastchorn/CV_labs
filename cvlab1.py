from PIL import Image
import numpy as np
# Відкриття зображення
input_image_path = "ph01.jpg"
output_image_path = "чб_зображення.jpg"
output_image_path_binary='бінарізоване_зображення.jpg'
image = Image.open(input_image_path)

# Розміри зображення
width, height = image.size

#Створєемо нове зображення у режимі градаціі сірого
bw_image = Image.new("L", (width, height))

# Проходимо кожен піксель і обчислюємо його яскравість
for x in range(width):
    for y in range(height):
        pixel_color = image.getpixel((x, y))
        #  Обчислення яскравості (градацій сірого) пікселя
        grayscale = int(0.36 * pixel_color[0] + 0.53 * pixel_color[1] + 0.11 * pixel_color[2])
        #grayscale= int((pixel_color[0] +  pixel_color[1] + pixel_color[2])/3) простийварік
        bw_image.putpixel((x, y), grayscale)


# Збереження чорно-білого зображення
bw_image.save(output_image_path)

#Перетворення зображення в масив NumPy для зручності використання
bw_array = np.array(bw_image)

# Обчислення гістограму яскравості
histogram = np.histogram(bw_array, bins=256, range=(0, 256))[0]

#Рохрахуємо порогове значення за Методом Крістіана
total = bw_array.size
threshold = 0
for t in range(1, 255):
    w0 = sum(histogram[:t]) / total
    w1 = sum(histogram[t:]) / total
    u0 = sum(i * histogram[i] for i in range(t)) / sum(histogram[:t])
    u1 = 0
    if sum(histogram[t:]) != 0:
     u1 = sum(i * histogram[i] for i in range(t, 256)) / sum(histogram[t:])
    var = w0 * w1 * (u0 - u1) ** 2
    if var > threshold:
     threshold = var
     best_threshold = t

# Бінаризуємо зображення, використовуючи порогове значення
binary_image = (bw_array > best_threshold) * 255

# Створюємо зображення з масива NumPy
binary_image = Image.fromarray(binary_image.astype(np.uint8))

# Зберігаємо бінарізоване зображення
binary_image.save(output_image_path_binary)

#Закриваємо зображення
bw_image.close()


# Вирізаємо об'єкт за допомогою маски
object_image = Image.new("RGB", image.size)
for x in range(image.width):
    for y in range(image.height):
        if binary_image.getpixel((x, y)) == 255:  # Пиксель у масці білий (об'єкт)
            pixel_color = image.getpixel((x, y))
            object_image.putpixel((x, y), pixel_color)

# Зберігаємо об'єкт
object_image.save("вирізанній_об'єкт.jpg")

# Закриваємо зображення
image.close()
binary_image.close()
object_image.close()
