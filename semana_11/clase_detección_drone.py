from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO('yolo11n-obb.pt')
    video = 'video.mp4'
    cap = cv2.VideoCapture(video)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        
        results = model(frame, verbose=False, conf=0.1)

        annoted_frame = results[0].plot()

        cv2.imshow("Video", annoted_frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()