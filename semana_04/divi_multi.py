import cv2
import numpy as np

# Cargar dos imágenes que deben tener el mismo tamaño y tipo
img1 = cv2.imread('imagen1.jpg').astype(np.float32)
img2 = cv2.imread('imagen2.jpg').astype(np.float32)

# Multiplicación de imágenes (pixel a pixel)
img_mult = cv2.multiply(img1, img2)

# División de imágenes (pixel a pixel), evitar división por cero
img2_safe = img2.copy()
img2_safe[img2_safe == 0] = 1  # para evitar división por cero
img_div = cv2.divide(img1, img2_safe)

# Convertir a uint8 para mostrar o guardar imágenes
img_mult = np.clip(img_mult, 0, 255).astype(np.uint8)
img_div = np.clip(img_div, 0, 255).astype(np.uint8)

# Mostrar resultados
cv2.imshow('Multiplicacion', img_mult)
cv2.imshow('Division', img_div)
cv2.waitKey(0)
cv2.destroyAllWindows()
