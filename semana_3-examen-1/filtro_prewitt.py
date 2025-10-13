import cv2
import numpy as np

img = cv2.imread('images.jpeg', 0)

#En 
kernel_prewitt_x = np.array([[-1, 0 , 1],
                            [-1, 0 , 1],
                            [-1, 0 , 1]])


kernel_prewitt_y = np.array([[1, 1 , 1],
                             [-1, 0 , 1],
                             [-1,-1 , -1]])

prewitt_x = cv2.filter2D(img, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(img, -1, kernel_prewitt_y)
prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

cv2.imshow('Prewitt', prewitt)
cv2.waitKey(0)