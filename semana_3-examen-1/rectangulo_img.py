import cv2

img = cv2.imread('images.jpeg')
img = cv2.rectangle(img, (0,0), (224,224), (0,255,255), 3)

cv2.imshow('Rectangulo', img)
cv2.waitKey(0)
