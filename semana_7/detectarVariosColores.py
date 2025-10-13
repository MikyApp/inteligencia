import cv2
import numpy as np

def dibujar(mask, color, nombre):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, nombre , (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)

            # Usar color pasado como argumento para el contorno
            cv2.drawContours(frame, [nuevoContorno], -1, color, 3)



ulr = 'http://192.168.5.221:4747/video'
cap = cv2.VideoCapture(ulr)

#Definir colores a detectar
#Azull
azulBajo = np.array([100, 100, 20], np.uint8)
azulAlto = np.array([125, 255, 255], np.uint8)

#Amarillo
amarilloBajo = np.array([28, 100, 20], np.uint8)
amarilloAlto = np.array([31, 255, 255], np.uint8)

#Rojo
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([5, 255, 255], np.uint8)
redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

while True:
    ret, frame = cap.read()

    if not ret:
        print('No se puede captura!')
        break

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
    maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
    maskRed1 =cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 =cv2.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1, maskRed2)

    azul = (255,0,0)
    amarillo = (0,255,255)
    rojo = (0,0,255)

    dibujar(maskAzul, azul, 'azul')
    dibujar(maskAmarillo, amarillo, 'amarillo')
    dibujar(maskRed, rojo, 'rojo')

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()


