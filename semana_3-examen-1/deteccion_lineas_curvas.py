import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar imagen y convertir a escala de grises
imagen = cv2.imread('images.jpeg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(gris, 50, 150, apertureSize=3)

# Detección de líneas con Transformada de Hough
lineas = cv2.HoughLines(bordes, 1, np.pi/180, 150)
imagen_lineas = np.copy(imagen)
if lineas is not None:
    for linea in lineas:
        rho, theta = linea[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Detección de círculos con Transformada de Hough
circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)
imagen_circulos = np.copy(imagen)
if circulos is not None:
    circulos = np.uint16(np.around(circulos))
    for i in circulos[0, :]:
        cv2.circle(imagen_circulos, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(imagen_circulos, (i[0], i[1]), 2, (0, 0, 255), 3)

# Mostrar resultados
titles = ['Original', 'Líneas detectadas', 'Círculos detectados']
images = [imagen, imagen_lineas, imagen_circulos]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
