import numpy as np
import cv2

img = np.zeros((224, 224, 3), np.uint8)
img = cv2.circle(img, (112,112), 100, (255,255,255),-1)
#cv2.imshow('plantilas', img)

img1 = cv2.imread('images.jpeg')


img3 = cv2.subtract(img, img1)

print(img1[0,0])
print(img3[0,0])

cv2.imshow('Resta ', img3)
cv2.waitKey(0)  

