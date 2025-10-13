import cv2
import numpy as np

cap = cv2.VideoCapture('http://192.168.5.221:4747/video')

#Rojo
bajoRojo = np.array([173,100,20])
altoRojo = np.array([179,255,255])

#Amarillo
while True:

    retm, frame = cap.read()
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Delimitar colores
    maskRojo = cv2.inRange(frameHSV, bajoRojo, altoRojo)

    kernel = np.ones((7,7), np.uint8)
    
    #Color rojo
    erosion = cv2.erode(maskRojo, kernel, iterations=2)
    dilatacion = cv2.dilate(erosion, kernel, iterations=2)
    x, y, w, h = cv2.boundingRect(dilatacion)

    #
    font = cv2.FONT_HERSHEY_COMPLEX
    #Colore rojo
    cv2.putText(frame, "Rojo", (x,y-12), font, 0.65, (255,0,0),1,cv2.LINE_AA)
    cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255),2)

    
    cv2.imshow('1', frame)
    
    k=cv2.waitKey(1)
    if k==27:
        break


cv2.destroyAllWindows()





