import cv2

img = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

#Aplicamos filtro de media
filtro_media = cv2.blur(img, (5,5))

cv2.imshow('Media', img)
cv2.waitKey(0)