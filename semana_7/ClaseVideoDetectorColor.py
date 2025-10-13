import cv2
import numpy as np

#img = cv2.imread('frutas.jpg')
cap = cv2.VideoCapture('http://192.168.5.221:4747/video')

def nothing(x):
    pass
cv2.namedWindow("BAR123")

#Tono
cv2.createTrackbar('Hmin','BAR123',0,179,nothing)
cv2.createTrackbar('Hmax','BAR123',0,179,nothing) 

#Pureza
cv2.createTrackbar('Smin','BAR123',0,255,nothing)
cv2.createTrackbar('Smax','BAR123',0,255,nothing)

#Luminosidad
cv2.createTrackbar('Vmin','BAR123',0,255,nothing)   
cv2.createTrackbar('Vmax','BAR123',0,255,nothing) 

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hMin = cv2.getTrackbarPos('Hmin','BAR123')
    hMax = cv2.getTrackbarPos('Hmax','BAR123')

    sMin = cv2.getTrackbarPos('Smin','BAR123')
    sMax = cv2.getTrackbarPos('Smax','BAR123')

    vMin = cv2.getTrackbarPos('Vmin','BAR123')
    vMax = cv2.getTrackbarPos('Vmax','BAR123')

    lower = np.array([hMin,sMin,vMin])
    upper = np.array([hMax,sMax,vMax])

    mask = cv2.inRange(hsv,lower, upper)
    
    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations=2)
    dilation = cv2.dilate(erosion,kernel,iterations=2)
    x,y,w,h = cv2.boundingRect(dilation)

    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(frame, "Objeto", (x,y-12), font, 0.65, (255,0,0),1,cv2.LINE_AA)
    cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),2)
    cv2.imshow('1', frame)

    
    k=cv2.waitKey(1)
    if k==27:
        break


cv2.destroyAllWindows()

"""
Entonces:

H = 347 / 2 ≈ 173 (en OpenCV)

S = 44% de 255 ≈ 112

V = 93% de 255 ≈ 237

Para definir el rango para detectar este color en OpenCV podrías probar:

Hmin = 165

Hmax = 179 (como máximo 179)

Smin = 100

Smax = 130 (un rango alrededor de 112 para permitir variantes)

Vmin = 220

Vmax = 255

"""