import cv2 as cv
from ultralytics import YOLO

def setup_model():

    model = YOLO("runs/pose/train8/weights/best.pt")
    #results = model.train(data = "hand-keypoints.yaml", epochs = 10, imgsz = 640, device = 0)

    return model


def start_video(model):

    cap = cv.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        results = model(frame, stream = True)
    
        for r in results:
            
            """
            Results is a generator. Typically, though, we only have one thing being detected so in this case we only care about r (the one hand).
            If we take this one hand, and go through its keypoints (a Tensor) we can find each xy (coordinate). With keypoints.xy, we get a tensor of 
            shape [0, 21, 2] which means that for each 21 keypoints, we have a tensor of 2 with x and y coordinates.
            """
            
            #print(r.keypoints.xy[0][0])
            
            """
            Now we also need the position of the whole hand. We can use the xyxy attribute to get two opposite corners of the box.
            """
            
            #print(r.boxes.xyxy[0][0], r.boxes.xyxy[0][1], r.boxes.xyxy[0][2], r.boxes.xyxy[0][3])
            #print(r.boxes.xyxy)
            
            bbox = r.boxes.xyxy.cpu().numpy()
            
            if bbox.shape[0] > 0:
                
                x_coords = [bbox[0][0], bbox[0][2]]
                y_coords = [bbox[0][1], bbox[0][3]]
                
                mid = (sum(x_coords) / 2.0, sum(y_coords) / 2.0)
                
                print(mid)
        
            
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