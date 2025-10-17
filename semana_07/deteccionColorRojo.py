"""
Hue / Matizm --> 179
Saturación / Saturación --> 255
Value / Brillo o valor --> 255
"""
import cv2
import numpy as np

ulr = 'http://192.168.5.221:4747/video'
cap = cv2.VideoCapture(ulr)

#Determinar rangos a detectar los colores
redBajo1 = np.array([0,100,20], np.uint8)
redAlto1 = np.array([8,255,255], np.uint8)

redBajo2 = np.array([173,100,20], np.uint8)
redAlto2 = np.array([179,255,255], np.uint8)

while True:
    ret, frame = cap.read()
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)

    maskRed = cv2.add(maskRed1, maskRed2)

    maskRedvis = cv2.bitwise_and(frame, frame, mask=maskRed)

    #Visualización de la detección
    cv2.imshow('maskRedvis', maskRedvis)
    cv2.imshow('maskRed', maskRed)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
