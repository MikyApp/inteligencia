from ultralytics import YOLO
import cv2

model = YOLO("bestnano.pt")
url = 'http://192.168.5.221:4747/video'
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        cls_id = int(box.cls)
        class_name = results[0].names[cls_id]
        confidence = box.conf

        # Acción para clase específica, por ejemplo "clase_1"
        if class_name == "abiertos" and confidence > 0.4:
            # Aquí pones el código que quieras ejecutar
            print(f"Detectado ojos abiertos")

        # Acción para clase específica, por ejemplo "clase_1"
        if class_name == "cerrados" and confidence > 0.4:
            # Aquí pones el código que quieras ejecutar
            print(f"Detectado ojos cerrados")
        
        # Acción para clase específica, por ejemplo "clase_1"
        if class_name == "bostezo" and confidence > 0.4:
            # Aquí pones el código que quieras ejecutar
            print(f"Detectado bostezo")
    

    cv2.imshow('Yolo inference', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
