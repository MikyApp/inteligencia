import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar imagen en escala de grises
imagen = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Filtro de Media (Promedio)
filtro_media = cv2.blur(imagen, (5, 5))

# Filtro Gaussiano
filtro_gaussiano = cv2.GaussianBlur(imagen, (5, 5), 0)

# Filtro Laplaciano
laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
laplaciano = cv2.convertScaleAbs(laplaciano)

# Filtro Prewitt (no está en OpenCV, se crea con kernels manualmente)
kernel_prewitt_x = np.array([[ -1, 0, 1],
                             [ -1, 0, 1],
                             [ -1, 0, 1]])
kernel_prewitt_y = np.array([[ 1, 1, 1],
                             [ 0, 0, 0],
                             [-1,-1,-1]])

prewitt_x = cv2.filter2D(imagen, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(imagen, -1, kernel_prewitt_y)
prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

# Filtro Sobel
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))

# Mostrar resultados con matplotlib (opcional)
titles = ['Imagen Original', 'Media', 'Gaussiano', 'Laplaciano', 'Prewitt', 'Sobel']
images = [imagen, filtro_media, filtro_gaussiano, laplaciano, prewitt, sobel]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]); plt.yticks([])
    

plt.show()
