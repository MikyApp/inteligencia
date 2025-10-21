import cv2
import mediapipe as mp

# Inicializa FaceMesh y los dibujos
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configura FaceMesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Para procesamiento en imágenes estáticas
    max_num_faces=2,        # Detectar solo una cara
    min_detection_confidence=0.5  # Confianza mínima de detección
) as face_mesh:
    
    # Abre el flujo de video desde la cámara IP
    cap = cv2.VideoCapture("http://192.168.5.105:4747/video")

    # Verifica que el flujo de video se haya abierto correctamente
    if not cap.isOpened():
        print("Error al abrir el flujo de video")
        exit()

    while cap.isOpened():
        # Lee un frame del flujo de video
        ret, img = cap.read()
        if not ret:
            print("No se pudo leer el frame.")
            break
        
        # Convierte la imagen de BGR (OpenCV) a RGB (requerido por Mediapipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Procesa la imagen con FaceMesh
        result = face_mesh.process(img_rgb)

        # Si se detectan marcas faciales, dibuja las marcas en la imagen
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, face_landmarks)

        # Muestra la imagen procesada
        cv2.imshow('Imagen', img)

        # Si se presiona la tecla 'q', sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera el objeto de captura de video y cierra las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()