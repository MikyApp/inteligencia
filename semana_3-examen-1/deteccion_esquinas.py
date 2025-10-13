import cv2
import numpy as np

# Cargar imagen en escala de grises
imagen = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Harris Corner Detection
imagen_harris = np.copy(imagen)
imagen_harris = cv2.cvtColor(imagen_harris, cv2.COLOR_GRAY2BGR)
harris = cv2.cornerHarris(imagen, 2, 3, 0.04)
harris = cv2.dilate(harris, None)
imagen_harris[harris > 0.01 * harris.max()] = [0, 0, 255]  # Marcar esquinas en rojo

# Shi-Tomasi Corner Detection
imagen_shitomasi = np.copy(imagen)
imagen_shitomasi = cv2.cvtColor(imagen_shitomasi, cv2.COLOR_GRAY2BGR)
corners = cv2.goodFeaturesToTrack(imagen, 100, 0.01, 10)
corners = np.int8(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(imagen_shitomasi, (x, y), 5, (0, 255, 0), -1)  # Marcar esquinas en verde

# Mostrar
from matplotlib import pyplot as plt

titles = ['Original', 'Harris', 'Shi-Tomasi']
images = [imagen, imagen_harris, imagen_shitomasi]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
