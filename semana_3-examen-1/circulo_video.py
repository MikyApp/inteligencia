import cv2

cap = cv2.VideoCapture("http://192.168.5.13:4747/video")

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    #Todo vamos a dibujar aqui
    frame_circulo = cv2.circle(frame, (320,240), 100, (0,0,255), 5)
    if not ret:
        print("Error: No se puede capturar el marco.")
    
    cv2.imshow('camara', frame_circulo)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
