import cv2
import numpy as np


def nada(x):
    pass

cv2.namedWindow('BAR123')

#Parametro H
cv2.createTrackbar('Hmin', 'BAR123', 0, 179, nada)
cv2.createTrackbar('Hmax', 'BAR123', 0, 179, nada)

#Parametro S
cv2.createTrackbar('Smin', 'BAR123', 0, 255, nada)
cv2.createTrackbar('Smax', 'BAR123', 0, 255, nada)

#Parametro V
cv2.createTrackbar('Vmin', 'BAR123', 0, 255, nada)
cv2.createTrackbar('Vmax', 'BAR123', 0, 255, nada)

#Kernel para suavizar 
cv2.createTrackbar('Kernel X', 'BAR123', 1, 30, nada)
cv2.createTrackbar('Kernel Y', 'BAR123', 1, 30, nada)

#Crear el video

cap = cv2.VideoCapture('http://192.168.5.221:4747/video')

while (1):
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        Hmin = cv2.getTrackbarPos('Hmin', 'BAR123')
        Hmax = cv2.getTrackbarPos('Hmax', 'BAR123')
    
        Smin = cv2.getTrackbarPos('Smin', 'BAR123')
        Smax = cv2.getTrackbarPos('Smax', 'BAR123')

        Vmin = cv2.getTrackbarPos('Vmin', 'BAR123')
        Vmax = cv2.getTrackbarPos('Vmax', 'BAR123')
    
        color_oscuro = np.array([Hmin, Smin, Vmin])
        color_brillante = np.array([Hmax, Smax, Vmax])

        mascara = cv2.inRange(hsv, color_oscuro, color_brillante)

        kernelx = cv2.getTrackbarPos('Kernel X', 'BAR123')
        kernely = cv2.getTrackbarPos('Kernel Y', 'BAR123')

        kernel = np.ones((kernelx, kernely), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contornos, -1, (0,0,0),2)
        cv2.imshow('camara', frame)
        cv2.imshow('Mascara', mascara)

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
        
cap.release()

