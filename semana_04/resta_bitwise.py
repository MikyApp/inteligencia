import numpy as np

import cv2

mask1 = np.zeros((224, 224, 3), np.uint8)
mask2 = cv2.circle(mask1, (112,112), 100, (255,255,255),-1)

img1 = cv2.imread('images.jpeg')


img2 = cv2.bitwise_and(img1, mask2)

#print(img1[0,0])
#print(mask2[0,0])

cv2.imshow('Resultado', img2)
cv2.waitKey(0)  

