import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

guinos = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
) as face_mech:
    while True:
        ret, frame =cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        alto,ancho = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_mech.process(img_rgb)

        if resultados.multi_face_landmarks:
            for face_landmarks in resultados.multi_face_landmarks:
                punto145 = face_landmarks.landmark[145]
                punto159 = face_landmarks.landmark[159]

                x1=int(punto145.x * ancho)
                y1=int(punto145.y * alto)

                x2=int(punto159.x * ancho)
                y2=int(punto159.y * alto)
                cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                fuente=cv2.FONT_HERSHEY_SIMPLEX

                if(y1-y2)<25:
                    guinos=+1
                    cv2.putText(frame, f'Ojos cerrados {guinos} ', (10,50), fuente, 1, (0,0,0),1,cv2.LINE_AA)
                    
                else:
                    
                    cv2.putText(frame, f'Ojos abiertos', (10,50), fuente, 1, (0,0,0),1,cv2.LINE_AA)    

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, ))
        cv2.imshow('Rostro', frame)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()