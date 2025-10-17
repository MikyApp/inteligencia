import cv2

img = cv2.imread('images.jpeg')
img = cv2.circle(img,(112,112), 100, (0,0,255), 5)

cv2.imshow('Circulo', img)
cv2.waitKey(0)
