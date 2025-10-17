import cv2

img = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

#Filtro sobel
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))

cv2.imshow('Sobel', sobel)
cv2.waitKey(0)