import cv2 as cv
from ultralytics import YOLO

def setup_model():

    model = YOLO("yolo26n-pose.pt")
    results = model.train(data = "hand-keypoints.yaml", epochs = 3, imgsz = 320)

    return model


def start_video(model):

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


def main():
    
    model = setup_model()
    start_video(model)


if __name__ == "__main__":

    main()