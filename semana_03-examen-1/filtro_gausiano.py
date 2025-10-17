import cv2

img = cv2.imread('images.jpeg')

filtro_gausiano = cv2.GaussianBlur(img, (5,5), 0)

cv2.imshow('Filtro gausiano', filtro_gausiano)
cv2.waitKey(0)
