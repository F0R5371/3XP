import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

results = model.train(data = "hand-keypoints.yaml", epochs = 5, imgsz = 640)

def start_video():

    cap = cv.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        results = model(frame, stream = True)

        for r in results:
            detected = r.plot()
            cv.imshow("Webcam", detected)

        if not ret:
            print("Can't receive frame, exiting loop.")
            break

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

start_video()