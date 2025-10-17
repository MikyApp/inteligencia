import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error, no se puede captura el marco")

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()