import cv2
import numpy as np

cap = cv2.VideoCapture('http://192.168.5.221:4747/video')

# Rangos HSV para colores
bajoRojo1 = np.array([0, 100, 20])
altoRojo1 = np.array([8, 255, 255])
bajoRojo2 = np.array([173, 100, 20])
altoRojo2 = np.array([179, 255, 255])

bajoAmarillo = np.array([113, 100, 100])
altoAmarillo = np.array([122, 255, 255])

while True:
    ret, frame = cap.read()
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara para rojo (considerando dos rangos para el rojo que se encuentra al inicio y final del rango HSV)
    maskRojo1 = cv2.inRange(frameHSV, bajoRojo1, altoRojo1)
    maskRojo2 = cv2.inRange(frameHSV, bajoRojo2, altoRojo2)
    maskRojo = cv2.add(maskRojo1, maskRojo2)

    # Máscara para amarillo
    maskAmarillo = cv2.inRange(frameHSV, bajoAmarillo, altoAmarillo)

    kernel = np.ones((7, 7), np.uint8)

    # Procesar rojo
    maskRojo = cv2.erode(maskRojo, kernel, iterations=2)
    maskRojo = cv2.dilate(maskRojo, kernel, iterations=2)
    contornosRojo, _ = cv2.findContours(maskRojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornosRojo:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Rojo", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Procesar amarillo
    maskAmarillo = cv2.erode(maskAmarillo, kernel, iterations=2)
    maskAmarillo = cv2.dilate(maskAmarillo, kernel, iterations=2)
    contornosAmarillo, _ = cv2.findContours(maskAmarillo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornosAmarillo:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Amarillo", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Deteccion de colores', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
