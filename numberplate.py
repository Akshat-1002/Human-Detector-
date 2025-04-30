import cv2
import numpy as np
import pytesseract
from djitellopy import Tello
from collections import deque

# YOLO model paths
yolo_cfg = "models/yolov4.cfg"
yolo_weights = "models/yolov4.weights"
coco_names = "models/coco.names"
license_plate_cfg = "models/license_plate.cfg"
license_plate_weights = "models/license_plate.weights"

# Load YOLO models
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
plate_net = cv2.dnn.readNet(license_plate_weights, license_plate_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
plate_layer_names = plate_net.getLayerNames()
plate_output_layers = [plate_layer_names[i - 1] for i in plate_net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(coco_names, "r") as f:
    classes = f.read().strip().split("\n")

drone = Tello()
drone.connect()
drone.streamon()

def detect_humans(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == classes.index("person"):
                x, y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


def detect_license_plates(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    plate_net.setInput(blob)
    outputs = plate_net.forward(plate_output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores[np.argmax(scores)]
            if confidence > 0.5:
                x, y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                plate_img = frame[y:y+h, x:x+w]
                plate_text = recognize_plate(plate_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def recognize_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    return text.strip()


def main():
    try:
        while True:
            frame = drone.get_frame_read().frame
            frame = cv2.resize(frame, (960, 720))
            frame = detect_humans(frame)
            frame = detect_license_plates(frame)
            cv2.imshow("Drone View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
