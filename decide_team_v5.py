import cv2
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from ultralytics import YOLO

from PIL import Image
import argparse

# define the function that handles our processing thread
def process_video(model_path:str,video_source,pwm_gpio:int,show:bool=True,enable_motor:bool=False):

    global model
    font = cv2.FONT_HERSHEY_SIMPLEX
    queuepulls = 0.0
    detections = 0
    fps = 0.0
    qfps = 0.0
    # init video
    cap = cv2.VideoCapture(video_source)
    

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize the input queue (frames), output queue (out),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    img = None
    out = None
    model = YOLO('./yolov8n_saved_model/yolov8n_full_integer_quant.tflite', task="detect")

    # construct a child process *indepedent* from our main process of
    # execution
    p = Process(target=classify_frame, args=(img, inputQueue, outputQueue,))
    p.daemon = True
    p.start()
    time.sleep(10)

    # time the frame rate....
    timer1 = time.time()
    frames = 0
    queuepulls = 0
    timer2 = 0
    t2secs = 0

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            if queuepulls == 1:
                timer2 = time.time()
            # Capture frame-by-frame
            # frame = frame.array
            img = Image.fromarray(frame)
            # if the input queue *is* empty, give the current frame to
            # classify
            if inputQueue.empty():
                inputQueue.put(frame)

            # if the output queue *is not* empty, grab the detections
            if not outputQueue.empty():
                out = outputQueue.get()

            if out is not None:
              for detected in out: 

                boxes = detected.boxes

                for box in boxes:

                    x1, y1, x2, y2 = box.xyxy[0]

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if int(box.cls[0]) == 0 and len(frame[y1:y2,x1:x2]) > 0:

                        person_arr = np.array([x1,y1,x2,y2,box.conf[0].cpu().numpy()]) #get data about detection into format required by tracker

                        people = np.vstack((people,person_arr))

            
            # FPS calculation
            frames += 1
            if frames >= 1:
                end1 = time.time()
                t1secs = end1-timer1
                fps = round(frames/t1secs, 2)
            if queuepulls > 1:
                end2 = time.time()
                t2secs = end2-timer2
                qfps = round(queuepulls/t2secs, 2)

        # Break the loop
        else:
            break

    p.join()
    # Everything done, release the vid
    cap.release()

    cv2.destroyAllWindows()

def classify_frame(img, inputQueue, outputQueue):
    global model
    global confThreshold
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue
            img = inputQueue.get()
            objs = model(img)
            outputQueue.put(objs)

import configparser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='path to cfg file', default="config.cfg")
    config = configparser.ConfigParser()

    # Load the configuration file
    args = parser.parse_args()
    config.read(args.cfg)
    modelPath = "/home/dennis/Desktop/Projects/Airsoft-player-tracking-project/yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
    camera_idx = 0
    confThreshold = 0.1
    pwm_gpio = 10
    show = True
    enable_motor = False
    process_video(model_path=modelPath,video_source=camera_idx,pwm_gpio=pwm_gpio,show=show,enable_motor=enable_motor)