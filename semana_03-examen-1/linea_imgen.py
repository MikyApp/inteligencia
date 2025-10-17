import cv2

img = cv2.imread('images.jpeg')
img = cv2.line(img, (0,0), (224,224), (0,0,255), 5)

cv2.imshow('Liena a imagen', img)
cv2.waitKey(0)
