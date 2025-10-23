from ultralytics import YOLO
import cv2 
import os

if __name__=="__main__":
    model = YOLO('yolo11n.pt')
    video = ''
    ulr = "http://192.168.137.55:4747/video"

    cap = cv2.VideoCapture(ulr)

    while True:
        ret, frame=cap.read()
        if ret is False:
            break

        results = model(frame, verbose = False)
        annoted_frame = results[0].plot()
        frameresize = cv2.resize(annoted_frame,(960,720))
        cv2.imshow("Video", frameresize)

        k = cv2.waitKey(1)

        if k == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    