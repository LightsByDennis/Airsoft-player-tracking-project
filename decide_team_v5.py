import multiprocessing as mp
import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO('./yolov8n_saved_model/yolov8n_full_integer_quant.tflite') #TFLOW model

# Define functions for each task

def capture_frames(queue):
    # Your frame capture logic here
    while True:
        vcap = cv.VideoCapture(0)  # Example function to capture frames
        frame = vcap.read()
        queue.put(frame)

def detect_objects(frame_queue, detection_queue):
    # Your object detection logic here
    while True:
        frame = frame_queue.get()
        detections = model(frame,stream=True) #detect objects in frame trough neural network  # Example function for object detection
        detection_queue.put(detections)

def control_turret(detection_queue):
    # Your turret control logic here
    while True:
        detection = detection_queue.get()

        for detected in detection: 

            boxes = detected.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if int(box.cls[0]) == 0:

                    person_arr = np.array([x1,y1,x2,y2,box.conf[0].cpu().numpy()]) #get data about detection into format required by tracker

                    people = np.vstack((people,person_arr))

        # Process detections and control turret accordingly

if __name__ == '__main__':
    # Create queues for inter-process communication
    frame_queue = mp.Queue()
    detection_queue = mp.Queue()

    # Create processes for each task
    frame_process = mp.Process(target=capture_frames, args=(frame_queue,))
    detection_process = mp.Process(target=detect_objects, args=(frame_queue, detection_queue))
    turret_process = mp.Process(target=control_turret, args=(detection_queue,))

    # Start processes
    frame_process.start()
    detection_process.start()
    turret_process.start()

    # Join processes
    frame_process.join()
    detection_process.join()
    turret_process.join()