from ultralytics import YOLO
import cv2
import os

if __name__ == " main ":
    model = YOLO('yolo11n-seg.pt') # Importando el modelo de deteccion
    video = 0 # "aeropuerto.mp4" # El video a procesar.
    cap = cv2.VideoCapture(video) # Cargar el video

while True:
    ret, frame = cap.read() # Leemos los frames del video
    if ret is False:
        break
    # Aplicamos el procesamiento al video
    results = model(frame, verbose=False, classes=[0,67], conf=0.5)

    # Dibujamos los resultados al video
    annoted_frame = results[0].plot()

    cv2.imshow("Video", annoted_frame) # Mostramos el video

    k = cv2.waitKey(1) # Insertamos una tecla
    
