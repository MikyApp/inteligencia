import cv2
import numpy as np

# Crear máscara circular rellena en escala de grises, tamaño igual al frame esperado (puedes ajustar)
mask = np.zeros((224, 224), np.uint8)  # Cambia 480x640 al tamaño real del frame
cv2.circle(mask, (112, 112), 100, (255,255,255), -1)  # Centro y radio ajustados

cap = cv2.VideoCapture("http://192.168.137.238:4747/video")

while True:
    ret, frame = cap.read()
    
    #Cambiar el tamano del frame
    #Por defecto trae 640, 480
    frame_modificado = cv2.resize(frame, (224, 224))

    # Aplicar máscara bitwise_and usando el canal máscara correcto
    masked_frame = cv2.bitwise_and(frame_modificado, frame_modificado, mask=mask)

    cv2.imshow('Camara con máscara', masked_frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
