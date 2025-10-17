import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar imagen en escala de grises
imagen = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Sobel
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))

# Prewitt (con kernels manuales)
kernel_prewitt_x = np.array([[ -1, 0, 1],
                             [ -1, 0, 1],
                             [ -1, 0, 1]])
kernel_prewitt_y = np.array([[ 1, 1, 1],
                             [ 0, 0, 0],
                             [-1,-1,-1]])
prewitt_x = cv2.filter2D(imagen, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(imagen, -1, kernel_prewitt_y)
prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

# Canny Edge Detector
canny = cv2.Canny(imagen, 100, 200)  # Umbrales bajos y altos (pueden ajustarse)

# Mostrar resultados
titles = ['Original', 'Sobel', 'Prewitt', 'Canny']
images = [imagen, sobel, prewitt, canny]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]); plt.yticks([])

plt.show()
