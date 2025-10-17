import cv2
import numpy as np


cap = cv2.VideoCapture('http://192.168.5.221:4747/video')

azulBajo = np.array([100, 100, 20], np.uint8)
azulAlto = np.array([125, 255, 255], np.uint8)

while True:
    ret, frame = cap.read()

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frameHSV, azulBajo, azulAlto)

    # Obtener los contornos (corregido el unpacking)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        #Solo capta areas azules mayores a 3000
        if area > 3000:
            #Dibuja cento y coordenadas
            M = cv2.moments(c)
            if (M["m00"]==0): M["m00"]=1
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])
            cv2.circle(frame, (x, y), 7, (0,255,0), -1)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, '{},{}'.format(x,y),(x+10,y), font, 0.75, (0,255,0),1,cv2.LINE_AA)

            #Se adapata mejor a los colores azules
            nuevoContorno = cv2.convexHull(c)
            cv2.drawContours(frame, [nuevoContorno], -1, (255, 0, 0), 3)

    # Visualizar resultados
    cv2.imshow('masAzul', mask)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
