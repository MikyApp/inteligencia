import cv2

#Leer las imagenes
imagen_1 = cv2.imread('azulejo.png')
imagen_2 = cv2.imread('corazon.jpg')

#Ver las dimensiones de las imagenes
ancho, alto = imagen_1.shape[:2]
ancho1, alto1 = imagen_2.shape[:2]
print(ancho, alto)
print(ancho1, alto1)

imagen_2_redimensionado = cv2.resize(imagen_2, (245, 305))

resta = cv2.subtract(imagen_1, imagen_2_redimensionado)

cv2.imshow('Diferencia', resta)
cv2.waitKey(0)

