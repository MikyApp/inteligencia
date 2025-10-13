import cv2
import numpy as np

fbgd = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture('video.mp4')  # o URL de la cámara IP
punto1 = [100, 600]
punto2 = [270, 450]
punto3 = [530, 350]
punto4 = [700, 600]
cont = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    height, width = frame.shape[:2]

    area_pts = np.array([punto1, punto2, punto3, punto4])
    # Ajustar puntos según el escalado para que coincide con el frame redimensionado
    scale_x, scale_y = 0.2, 0.2
    area_pts_scaled = np.array([[int(x * scale_x), int(y * scale_y)] for (x, y) in area_pts])

    frame = cv2.drawContours(frame, [area_pts_scaled], -1, (0, 0, 255), 3)

    fgmask = fbgd.apply(frame)
    imgAux = np.zeros((height, width), dtype=np.uint8)

    cv2.drawContours(imgAux, [area_pts_scaled], -1, (255, 255, 255), -1)
    fgmask = cv2.bitwise_and(fgmask, fgmask, mask=imgAux)

    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        if cv2.contourArea(cnt) > 1000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Objeto Detectado', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            # Aplicamos contador para el centro del objeto
            if 450 * scale_y <= y + h / 2 <= 460 * scale_y:
                cont += 1

    cv2.putText(frame, f'Personas detectados: {cont}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Imagen Auxiliar', imgAux)
    cv2.imshow('Mascara', fgmask)

    k = cv2.waitKey(1)
    if k == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
