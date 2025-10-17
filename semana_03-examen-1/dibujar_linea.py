import numpy as np
import cv2

img = np.zeros((512,512, 3), np.uint8)
img = cv2.line(img, (0,0), (511,511), (0,255,255), 5)
cv2.imshow('Linea', img)
cv2.waitKey(0)