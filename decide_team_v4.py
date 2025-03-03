import cv2 as cv

import numpy as np

from ultralytics import YOLO

from packages import sort, turret

from threading import Thread

import time, tkinter

from time import sleep

import time

customWidth = 640
customHeight = 480

enemy_teams = ['red']

calibrationTimeLimit = 5

print("===============================================================================================================")
print("__/\\\\\\\\\\\\\\\\\\\\\\\\_______________________________________________/\\\\\\_________                                    ")
print(" _\/\\\\\\////////\\\\\\____________________________________________\\/\\\\\\_________                                   ")
print("  _\/\\\\\\______\//\\\\\\___________________/\\\\\\____________________\/\\\\\\_________                                  ")
print("   _\/\\\\\\_______\/\\\\\\__/\\\\\\____/\\\\\\__/\\\\\\\\\\\\\\\\\\\\\\_____/\\\\\\\\\\\\\\\\_\/\\\\\\_________                                 ")
print("    _\/\\\\\\_______\/\\\\\\_\/\\\\\\___\/\\\\\\_\////\\\\\\////____/\\\\\\//////__\/\\\\\\\\\\\\\\\\\\\\__                                ")
print("     _\/\\\\\\_______\/\\\\\\_\/\\\\\\___\/\\\\\\____\/\\\\\\_______/\\\\\\_________\/\\\\\\/////\\\\\\_                               ")
print("      _\/\\\\\\_______/\\\\\\__\/\\\\\\___\/\\\\\\____\/\\\\\\_/\\\\__\//\\\\\\________\/\\\\\\___\/\\\\\\_                              ")
print("       _\/\\\\\\\\\\\\\\\\\\\\\\\\/___\//\\\\\\\\\\\\\\\\\\_____\//\\\\\\\\\\____\///\\\\\\\\\\\\\\\\_\/\\\\\\___\/\\\\\\_                             ")
print("        _\////////////______\/////////_______\/////_______\////////__\///____\///__                            ")
print("____/\\\\\\\\\\\\\\\\\\___________________________/\\\\\\__________/\\\\\\____________________________________________        ")
print(" __/\\\\\\///////\\\\\\________________________\/\\\\\\_________\/\\\\\\____________________________________________       ")
print("  _\/\\\\\\_____\/\\\\\\________________________\/\\\\\\_________\/\\\\\\____________________________________________      ")
print("   _\/\\\\\\\\\\\\\\\\\\\\\\/_____/\\\\\\____/\\\\\\________\/\\\\\\_________\/\\\\\\______/\\\\\\\\\\\\\\\\___/\\\\/\\\\\\\\\\\\\\___/\\\\\\\\\\\\\\\\\\\\_     ")
print("    _\/\\\\\\//////\\\\\\____\/\\\\\\___\/\\\\\\___/\\\\\\\\\\\\\\\\\\____/\\\\\\\\\\\\\\\\\\____/\\\\\\/////\\\\\\_\/\\\\\\/////\\\\\\_\/\\\\\\//////__    ")
print("     _\/\\\\\\____\//\\\\\\___\/\\\\\\___\/\\\\\\__/\\\\\\////\\\\\\___/\\\\\\////\\\\\\___/\\\\\\\\\\\\\\\\\\\\\\__\/\\\\\\___\///__\/\\\\\\\\\\\\\\\\\\\\_   ")
print("      _\/\\\\\\_____\//\\\\\\__\/\\\\\\___\/\\\\\\_\/\\\\\\__\/\\\\\\__\/\\\\\\__\/\\\\\\__\//\\\\///////___\/\\\\\\_________\////////\\\\\\_  ")
print("       _\/\\\\\\______\//\\\\\\_\//\\\\\\\\\\\\\\\\\\__\//\\\\\\\\\\\\\\/\\\\_\//\\\\\\\\\\\\\\/\\\\__\//\\\\\\\\\\\\\\\\\\\\_\/\\\\\\__________/\\\\\\\\\\\\\\\\\\\\_ ")
print("        _\///________\///___\/////////____\///////\//___\///////\//____\//////////__\///__________\//////////__")
print("===============================================================================================================")
sleep(1)
print("A-Cat V1.0")
print("Automated - Cunt anihilation turret")
print("Initialising Systems....")
sleep(1.5)
print("Systems Initialised")
print("Running diagnostics....")
sleep(0.5)
print("System integrity at 100%")
sleep(0.2)
print("Booting up Neural-Network")
sleep(1.5)
print("Systems operational")
print("Starting Calibration....")
print("===============================================================================================================")

# =================================================================================

# Setting up Stepper Motor functions

# =================================================================================



import RPi.GPIO as GPIO



Step1Return = 600   # How many steps to get back to center on motor 1

Step2Return = 600   # How many steps to get back to center on motor 2



RPM = 60

PulsePerRotation = 800

PulseSleep = (1 / ((RPM / 60) * PulsePerRotation)) / 2



Enable = 40

Dir1 = 7

Step1 = 12

Dir2 = 18

Step2 = 23

CW = 1

CCW = 0

firstRun = 1



GPIO.setmode(GPIO.BOARD)



# Establish Pins in software

GPIO.setup(Dir1, GPIO.OUT)

GPIO.setup(Step1, GPIO.OUT)

GPIO.setup(Dir2, GPIO.OUT)

GPIO.setup(Step2, GPIO.OUT)

GPIO.setup(Enable, GPIO.OUT)



# Setting initial output Low

GPIO.setup(Dir1, GPIO.LOW)

GPIO.setup(Dir2, GPIO.LOW)

GPIO.setup(Step1, GPIO.LOW)

GPIO.setup(Step2, GPIO.LOW)

GPIO.setup(Enable, GPIO.HIGH)



# =================================================================================

# Calibrate to home position

# =================================================================================


start_time = time.time()

try:

    while firstRun == 1:

        current_time = time.time()



    # Calculate the elapsed time

        elapsed_time = current_time - start_time



    # Exit the loop if the time limit is reached

        if elapsed_time >= calibrationTimeLimit:

            break



        print("Calibrating CIWS")

        GPIO.output(Dir1, CW)

        GPIO.output(Dir2, CW)



        sleep(PulseSleep)



        print("Stepping Motor 1")

        GPIO.output(Step1, GPIO.HIGH)

        GPIO.output(Step1, GPIO.LOW)



        print("Stepping Motor 2")

        GPIO.output(Step2, GPIO.HIGH)

        GPIO.output(Step2, GPIO.LOW)



except:

    print("error")



sleep(3)



try:

    GPIO.output(Dir1, CCW)

    GPIO.output(Dir2, CCW)

    for x in range(0, Step1Return):

        print("Stepping motor 1 Back")

        sleep(PulseSleep * 2)

        GPIO.output(Step1, GPIO.HIGH)

        GPIO.output(Step1, GPIO.LOW)

    

    for x in range(0, Step1Return):

        print("Stepping motor 2 Back")

        sleep(PulseSleep * 2)

        GPIO.output(Step2, GPIO.HIGH)

        GPIO.output(Step2, GPIO.LOW)



except:

    print("Error part 2 Electric Boogaloo")



# =================================================================================

# Completed startup sequence

# =================================================================================



class WebcamStream : #credits to https://github.com/vasugupta9 (https://github.com/vasugupta9/DeepLearningProjects/blob/main/MultiThreadedVideoProcessing/video_processing_parallel.py)

    def __init__(self, stream_id=0): 

        self.stream_id = stream_id   # default is 0 for primary camera 

        

        # opening video capture stream 

        self.vcap      = cv.VideoCapture(self.stream_id)

        if self.vcap.isOpened() is False :

            print("[Exiting]: Error accessing webcam stream.")

            exit(0)

        fps_input_stream = int(self.vcap.get(5))

        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

            

        # reading a single frame from vcap stream for initializing 

        self.grabbed , self.frame = self.vcap.read()

        if self.grabbed is False :

            print('[Exiting] No more frames to read')

            exit(0)



        # self.stopped is set to False when frames are being read from self.vcap stream 

        self.stopped = True 



        # reference to the thread for reading next available frame from input stream 

        self.t = Thread(target=self.update, args=())

        self.t.daemon = True # daemon threads keep running in the background while the program is executing 

        

    # method for starting the thread for grabbing next available frame in input stream 

    def start(self):

        self.stopped = False

        self.t.start() 



    # method for reading next frame 

    def update(self):

        while True :

            if self.stopped is True :

                break

            self.grabbed , self.frame = self.vcap.read()

            if self.grabbed is False :

                print('[Exiting] No more frames to read')

                self.stopped = True

                break 

        self.vcap.release()



    # method for returning latest read frame 

    def read(self):

        return self.frame



    # method called to stop reading frames 

    def stop(self):

        self.stopped = True 



class Id_team(): #associate id with team

    def __init__(self,Id,team=None,all_teams=[],countdown=30,ttit=30) -> None:

        self.Id = Id

        self.team = team

        self.teams = all_teams

        self.teams_values = [ 0 for i in all_teams]

        self.all_teams = all_teams

        self.countdown = countdown # number of frames in row where id is not found, object gets deleted after reaching 0

        self.max_countdown = countdown

        self.ttit = ttit # "time to identify team" if colorless team is playing, this is cooldown till Unknown player is marked as colorless player

        self.max_ttit = ttit

        self.switched_to_colorless = False

    def update_team(self,team,colorless_playing):

        if self.ttit <= 0 and not self.switched_to_colorless:

            for t in self.teams:

                if t.name == 'Unknown':

                    self.teams[self.teams.index(t)] = self.all_teams[0]

        if team.name == 'Unknown':

            if colorless_playing:

                if not self.switched_to_colorless:

                    self.teams_values[self.teams.index(team)] += 1

                else:

                    self.teams_values[1] += 1

                self.team = self.teams[self.teams_values.index(max(self.teams_values))]

        elif self.team.name == 'Unknown' and not colorless_playing:

            self.team = team

            self.teams_values[self.teams.index(team)] += 1

        else:

            self.teams_values[self.teams.index(team)] += 1

            self.team = self.teams[self.teams_values.index(max(self.teams_values))]

        self.countdown = self.max_countdown



class Ids():

    def __init__(self,teams,colorless_playing=False) -> None:

        self.ids = []

        self.updated = []

        self.colorless_playing = colorless_playing

        self.teams = teams



    def get_id_from_ids(self,wanted_id):

        for id in self.ids:

            if id.Id == wanted_id:

                return id

        return None

            

    def check_id(self,id_to_check,team):

        id = self.get_id_from_ids(id_to_check)

        if id != None:

            id.update_team(team,self.colorless_playing)

            self.updated.append(id.Id)

            return id.team

        self.ids.append(Id_team(id_to_check,team,self.teams))

        self.updated.append(id_to_check)

        return team



    def update(self):

        to_pop = []

        for id in self.ids:

            if id.countdown <= 0:

                to_pop.append(id.Id)

            if id.Id not in self.updated:

                id.countdown -= 1

            if id.team.name == 'Unknown' and self.colorless_playing and id.ttit > 0:

                id.ttit -= 1

            elif id.team.name != 'colorless':

                id.ttit = id.max_ttit

        for id_to_pop in to_pop:

            self.ids.pop(self.ids.index(self.get_id_from_ids(id_to_pop)))

    



class Team(): #class containing info about what clolor range of armband is associated to which team name

    def __init__(self,name,upper_color,lower_color,display_color=(255,0,255)) -> None:

        self.name = name #team name

        self.upper_color = upper_color # brightest/highest color shade that is recognized as teams armband (numpy array, color has to be in VHS format)

        self.lower_color = lower_color # darkest/lowest color shade that is recognized as teams armband (numpy array, color has to be in VHS format)

        self.display_color = display_color # color of player border (mainly for debuging purposes)



def find_closest_enemy(enemies,screencenter):

    if len(enemies) > 0:

        centers = []

        for enemy in enemies:

            x1,y1,x2,y2,Id = enemy

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            center = [round(x1+abs(x1-x2)/2),round(y1+abs(y1-y2)/2)]

            centers.append(center)

        closest_center = centers[0]

        closest_center_dist = np.sqrt(abs(closest_center[0]-screencenter[0])**2+abs(closest_center[1]-screencenter[1])**2)

        for center in centers:

            if np.sqrt(abs(center[0]-screencenter[0])**2+abs(center[1]-screencenter[1])**2) < closest_center_dist:

                closest_center = center

        return closest_center, enemies[centers.index(closest_center)]



#model = YOLO('./Yolo weights/yolov8n.pt') #Base performance 1FPS
model = YOLO('./yolov8n_saved_model/yolov8n_full_integer_quant.tflite') #TFLOW model 3FPS
#model = YOLO('./yolov8n_saved_model/yolov8n_integer_quant.tflite') 700ms

tracker = sort.Sort(30,1)



colorless_playing = False # True = FORCE DETECTION OF COLORLESS TEAM !

people = np.empty((0,5))

color = (0,0,255)



capture = 0# <--- set video capture (source)



stream = WebcamStream(capture)

stream.start()



#width = int(stream.vcap.get(cv.CAP_PROP_FRAME_WIDTH ))
width = int(640)
#height = int(stream.vcap.get(cv.CAP_PROP_FRAME_HEIGHT ))
height = int(480)

screencenter = [round(width/2),round(height/2)]

screencenter2 = [round(width/2)-125,round(height/2)+230]

screencenter3 = [round(width/2)-125,round(height/2)+200]





all_teams = [ # \/ add/change teams  \/ --------------------------------------------------------

    Team('Unknown', np.array([0,0,0]), np.array([255,255,255]), (0,255,0)), #used for people not matching description of any other team, !-DO NOT CHANGE OR REMOVE-!



    Team('colorless', np.array([0,0,0]), np.array([255,255,255]), (255,0,255)),#special team with invalid color range, only if team with no color is playing



    Team('blue', np.array([123,255,191]), np.array([106,174,52]), (255,0,0)),

    Team('red', np.array([179,255,255]), np.array([162,169,106]), (0,0,255)),

    Team('yellow', np.array([29,255,255]), np.array([18,165,89]), (0,255,255))

] # You can add more teams, team object syntax: Team('name of team', brightest color of armband (in VHS format), lowest color of armband (also VHS))



playing_teams = ['red','blue'] # EDIT ALL PLAYING TEAMS !



teams = []

teams.append(all_teams[0])

for et in playing_teams:

    for t in all_teams:

        if t.name == et:

            teams.append(t)

if 'colorless' in playing_teams:

    colorless_playing = True

ids = Ids(teams,colorless_playing)



if not stream.vcap.isOpened():

    print("Cannot open camera")

    exit()

last_frame_time = time.time()





while True: # Main loop !!!!!!

    last_frame_time = time.time()



    frame = stream.read() #get frame from camera

    resizedFrame= cv.resize(frame, (customWidth, customHeight))

    detection = model(resizedFrame,stream=True) #detect objects in frame trough neural network



    frame_out = np.copy(frame)

    people = np.empty((0,5))



    #find people in detected objects

    for detected in detection: 

        boxes = detected.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if int(box.cls[0]) == 0 and len(frame[y1:y2,x1:x2]) > 0:

                person_arr = np.array([x1,y1,x2,y2,box.conf[0].cpu().numpy()]) #get data about detection into format required by tracker

                people = np.vstack((people,person_arr))



    tracker_return = tracker.update(people) #sends data about detections to sort, sort tryes to associate people from previous frames with new detections gives them IDs

    enemies = np.empty((0,5))

    for res in tracker_return:

        x1,y1,x2,y2,Id = res

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)



        center = [round(x1+abs(x1-x2)/2),round(y1+abs(y1-y2)/2)]



        person = frame[y1:y2,x1:x2]

        if len(person) > 0 and all(i > -1 for i in [x1,y1,x2,y2]): #check if cordinates of person are valid, othervise empty selections or negative cordinates can cause openCV error

            #find color matches with defined teams

            hsv_person = cv.cvtColor(person,cv.COLOR_BGR2HSV)

            mask_sums = []

            for team in teams:

                mask = cv.inRange(hsv_person,team.lower_color,team.upper_color)

                mask_sums.append(np.sum(mask))

            if max(mask_sums) > 15:

                best_team_match = teams[mask_sums.index(max(mask_sums))]

            else:

                best_team_match = all_teams[0]



            person_team = ids.check_id(Id,best_team_match)

            color = person_team.display_color



            if person_team.name in enemy_teams:

                enemies = np.vstack((enemies,res))

            

            # graphics for visual validation of data

            cv.rectangle(frame_out,(x1,y1),(x2,y2),color,2)

            cv.drawMarker(frame_out,center,color,cv.MARKER_CROSS,thickness=2)

            cv.putText(frame_out,person_team.name,np.array([x1+10,y1-10]),cv.FONT_HERSHEY_SIMPLEX,1,color,2,cv.LINE_AA)

            cv.putText(frame_out,str(int(Id)),np.array([x1,y2-10]),cv.FONT_HERSHEY_SIMPLEX,1,color,2,cv.LINE_AA)





            cv.drawMarker(frame_out,screencenter,(255,0,255),cv.MARKER_CROSS,50,2)



    if len(enemies) > 0:

        closest_center, closest_enemy = find_closest_enemy(enemies,screencenter)

        

        # Calculate distances

        distance_x = abs(closest_center[0] - screencenter[0])

        distance_y = abs(closest_center[1] - screencenter[1])

        distance_x2 = "Target X = "

        distance_y2 = "Target Y = "



        printdistance_x = distance_x2+str(closest_center[0])

        printdistance_y = distance_y2+str(closest_center[1])



        # Print distances

        print("Distance to screen center:")

        print("X-axis:", distance_x)

        print("Y-axis:", distance_y)



        cv.line(frame_out,closest_center,screencenter,(255,0,255),2,cv.LINE_AA)

        cv.drawMarker(frame_out,closest_center,ids.get_id_from_ids(closest_enemy[4]).team.display_color,cv.MARKER_SQUARE,thickness=2)

        cv.putText(frame_out,str(printdistance_x),screencenter3,cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv.LINE_AA)

        cv.putText(frame_out,str(printdistance_y),screencenter2,cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv.LINE_AA)

    

    ids.update()

    cv.imshow("test",frame_out)

    cv.waitKey(1)