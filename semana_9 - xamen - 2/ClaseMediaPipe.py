import cv2
import mediapipe as mp

mp_face_mesh=mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = 6,
    min_detection_confidence = 0.1
) as face_mesh:
    img = cv2.imread('varios.webp')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    print("Marcas Faciales: ", result.multi_face_landmarks)

    if result.multi_face_landmarks is not None:
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(img, face_landmarks)
    cv2.imshow('Imagen', img)

cv2.waitKey(0)
cv2.destroyAllWindows(0)