import cv2

img = cv2.imread('images.jpeg')

lapaciano = cv2.Laplacian(img, cv2.CV_64F)
lapaciano = cv2.convertScaleAbs(lapaciano)

cv2.imshow('Lapaciano', lapaciano)
cv2.waitKey(0)
