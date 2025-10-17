import cv2
import numpy as np

fbgd = cv2.createBackgroundSubtractorMOG2()  # RESTADOR DE FONDO CON IMAGEN
cap = cv2.VideoCapture('porf.mp4') #http://192.168.137.238:4747/video
punto1 = [0, 600]
punto2 = [0, 0]
punto3 = [800, 0]
punto4 = [800, 600]
cont = 0

while True:
    ret, frame = cap.read()  # ret=habilitado el video, frame=contenido
    if not ret:  # Si no se puede leer el video, se sale del bucle
        break
    
    frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)  # REDIMENSIONANDO la imagen 
    
    area_pts = np.array([punto1, punto2, punto3, punto4])  # Creando el área del filtro de la imagen
    frame = cv2.drawContours(frame, [area_pts], -1, (0, 0, 255), 3)  # Graficar en área a filtrar
    
    fgmask = fbgd.apply(frame)  # Aplicar el restador a la imagen 
    imgAux = np.zeros((600, 800), dtype=np.uint8)  # Creamos la imagen auxiliar para el área previa
    
    cv2.drawContours(imgAux, [area_pts], -1, (255, 255, 255), -1)  # Graficamos el contorno de la imagen auxiliar
    fgmask = cv2.bitwise_and(fgmask, fgmask, mask=imgAux)  # Aplicamos el producto de imágenes entre el restador y la máscara
    
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Detectamos contornos
    for cnt in cnts:
        if cv2.contourArea(cnt) > 1000:  # Aplicando un filtro de área mínima por cada contorno
            (x, y, w, h) = cv2.boundingRect(cnt)  # Ubicamos las posiciones del objeto detectado 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Graficamos un rectángulo en el objeto detectado
            cv2.putText(frame, 'Objeto Detectado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Ingresamos un texto de objeto detectado
            # Aplicando contador
            if 450 <= y + h / 2 <= 460:
                cont += 1
    
    cv2.putText(frame, f'personas detectadas: {cont}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Original', frame)  
    cv2.imshow('Imagen Auxiliar', imgAux)
    cv2.imshow('Mascara', fgmask)  # Mostramos la máscara
    
    k = cv2.waitKey(1)
    if k == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()