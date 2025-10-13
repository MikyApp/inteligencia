import cv2
import numpy as np

img = cv2.imread('frutas.jpg')

def nothing(x):
    pass

cv2.namedWindow('BAR123')

# Parámetro H
cv2.createTrackbar('Hmin', 'BAR123', 0, 179, nothing)
cv2.createTrackbar('Hmax', 'BAR123', 0, 179, nothing)

# Parámetro S
cv2.createTrackbar('Smin', 'BAR123', 0, 255, nothing)
cv2.createTrackbar('Smax', 'BAR123', 0, 255, nothing)

# Parámetro V
cv2.createTrackbar('Vmin', 'BAR123', 0, 255, nothing)
cv2.createTrackbar('Vmax', 'BAR123', 0, 255, nothing)

while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hMin = cv2.getTrackbarPos('Hmin', 'BAR123')
    hMax = cv2.getTrackbarPos('Hmax', 'BAR123')

    sMin = cv2.getTrackbarPos('Smin', 'BAR123')
    sMax = cv2.getTrackbarPos('Smax', 'BAR123')
    
    vMin = cv2.getTrackbarPos('Vmin', 'BAR123')
    vMax = cv2.getTrackbarPos('Vmax', 'BAR123')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7,7),np.uint8)
    
    erosion = cv2.erode(mask, kernel, iterations=2)
    dilatacion = cv2.dilate(erosion, kernel, iterations=2)

    x,y,w,h = cv2.boundingRect(dilatacion)

    img_copy = img.copy()
    
    cv2.rectangle(img_copy,(x,y),(x+w,y+h), (255,0,0),2)
    cv2.imshow('1', img_copy)

    k = cv2.waitKey(1)
    if k == 27:
        break


cv2.destroyAllWindows()
