from flask import Flask, send_file
from time import sleep
import RPi.GPIO as GPIO

 

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


#hier define je de GPIO pinnen

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

app = Flask('light')


GPIO.output(Dir1, CW)

GPIO.output(Dir2, CW)


@app.route('/')

def index():

       return send_file('light.html')

 

@app.route('/images/<filename')

def get_image(filename):

       return send_file('images/'+filename)

 

#als de server het turnOn signaal geeft gaat het lampje branden

@app.route('/turnOn')

def turnOn():

       for x in range(0, 5000):

            print("Stepping motor 1 Back")

            sleep(PulseSleep * 2)

            GPIO.output(Step1, GPIO.HIGH)

            GPIO.output(Step1, GPIO.LOW)

       return 'turnedOn'

 

#als de server het turnOff signaal geeft gaat het lampje uit    

@app.route('/turnOff')

def turnOff():

       return 'turnedOff'

 

app.run(debug=true, port=3000, host='0.0.0.0')