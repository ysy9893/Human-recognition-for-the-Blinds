#####################################################################################
#This code is based on "url link" 
#I newly added Non maximum suppression and tracking algorithm

#There're some minor modifications to video stream feeds. 
#Instead of creating video instances, I employed useful modules from imutils library 
#####################################################################################
#####################################################################################
## Import packages

import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from nms import NMS
from threading import Thread
from motpy import Detection, MultiObjectTracker
import requests
from speech import speak
from face_recog import face_recog, load_known_face

#####################################################################################
#####################################################################################
#Videostreaming using multi threading 
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(700, 700), framerate=30, src=0):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(src)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
#####################################################################################
#####################################################################################
##############################FACE_API & PRODUCT_API#################################
API_URL_face= 'https://dapi.kakao.com/v2/vision/face/detect'
API_URL_product='https://dapi.kakao.com/v2/vision/product/detect'
MYAPP_KEY = '4cdab2c7ed5f6af444ab8a3316bf914e'

##FUNCTIONS

def detect_face(filename):
    headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

    try:
        #files = { 'image' : filename}
        files={'image':open(filename, 'rb')}
        resp_face = requests.post(API_URL_face, headers=headers, files=files)
        resp_face.raise_for_status()

        return resp_face.json()
    except Exception as e:
        print(str(e))
        sys.exit(0)

def detect_product(filename):
    headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

    try:
        #files = { 'image' : filename}
        files={'image':open(filename, 'rb')}
        resp_product=requests.post(API_URL_product,headers=headers,files=files)
        resp_product.raise_for_status()

        return resp_product.json()
    except Exception as e:
        print(str(e))
        sys.exit(0)


def bbox(frame, detection_result_face,detection_result_product):

    for face,product in zip(detection_result_face['result']['faces'],detection_result_product['result']['objects']):
        x1_f = int(face['x']*imW)
        w_f=int(face['w']*imW)
        y1_f= int(face['y']*imH)
        h_f=int(face['h'] * imH)
        yaw=str(np.rad2deg(face['yaw']))
        gender=face['facial_attributes']['gender']
        age=str(face['facial_attributes']['age'])
        if float(gender['male'])>float(gender['female']):
            gender='Male'
        else:
            gender='Female'
        
        x1_p= int(product['x1'] *imW)
        y1_p= int(product['y1'] * imH)
        x2_p= int(product['x2'] * imW)
        y2_p= int(product['y2'] * imH)
        label=product['class']
        
        cv2.rectangle(frame, (x1_f, y1_f), (x1_f+w_f, y1_f+h_f),
                      (255, 255, 255), 2)
        cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p),
                      (0, 0, 0), 2)
        cv2.putText(frame, label, (x1_p, y1_p-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        cv2.putText(frame, gender+' '+age+' '+yaw, (x1_f, y1_f-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    
#####################################################################################
#####################################################################################
#Define and parse input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='700x700')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
#####################################################################################
#####################################################################################
## Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   
#####################################################################################
#####################################################################################
## Load the model & Labels

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
######################################################################################
######################################################################################
## Configuration of variables

#Initialize tensors
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()#Return clock cycle per second 
######################################################################################
######################################################################################
###Inferencing

##Load face_recogntion model
target_face_encoding=load_known_face('target.jpg')

## Initialize multiple video stream
webcam1=VideoStream(resolution=(700,700),framerate=30,src=0).start()
'''
webcam2=VideoStream(resolution=(700,700),framerate=30,src=2).start()
webcam3=VideoStream(resolution=(700,700),framerate=30,src=4).start()
'''

##Initialize Tracker 
tracker=MultiObjectTracker(dt=0.1) #100ms


##Test Speech
text="만나서 대단히 반갑습니다"
speak(text)
##Get video stream feeds 
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    #Grab frame from video stream

    frame1=webcam1.read()
    '''
    frame2=webcam2.read()
    frame3=webcam3.read()

    frame1=cv2.hconcat([frame1,frame2])
    frame1=cv2.hconcat([frame1,frame3])
    '''


    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)


    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()#inferencing 

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    
    ##Non Maximum Suppression 
    boxes,scores,classes=NMS(boxes,classes,scores,0.5,imH,imW)

    
    ############################# Configuration for Tracking ##########################################
    boxes=np.array(boxes)#current col order is [ymin,xmin,ymax,ymin]
    #Change the order of cols to [xmin,ymin,xmax,ymax] which is suitable feature of feed for tracker
    xmin=boxes[:,1]*imW
    xmin[xmin<1]=1
    xmin=xmin.reshape((-1,1))
    ymin=boxes[:,0]*imH
    ymin[ymin<1]=1
    ymin=ymin.reshape((-1,1))
    xmax=boxes[:,3]*imW
    xmax[xmax>imW]=imW
    xmax=xmax.reshape((-1,1))
    ymax=boxes[:,2]*imH
    ymax[ymax>imH]=imH
    ymax=ymax.reshape((-1,1))
    
    boxes=np.concatenate((xmin,ymin,xmax,ymax),axis=1)
    boxes=[i for idx,i in enumerate(boxes) if scores[idx]>min_conf_threshold and scores[idx]<=1.0]
    classes=[i for idx,i in enumerate(classes) if scores[idx]>min_conf_threshold and scores[idx]<=1.0]
    scores=[i for i in scores if i >min_conf_threshold and i<=1.0]
    
    ##Tracking 
    detections=[Detection(box=bbox,score=sc,cl=cl) for bbox, sc, cl in zip(boxes,scores,classes)]
    tracker.step(detections)
    tracks=tracker.active_tracks()
    

    ###################################################################################################
    #################################output######################################################
    #resize output frame
    frame=cv2.resize(frame,(imW,imH))
    ##############################Face & product recognition############################################################
    
    #ret,file=cv2.imencode('.jpg',frame)
    #file=file.tobytes()
    #file='./temp.jpg'
    #cv2.imwrite(file,frame)
    #detection_result_face = detect_face(file)
    #detection_result_product=detect_product(file)
    #bbox(frame, detection_result_face,detection_result_product)
    faces=face_recog(frame,target_face_encoding)
    for face in faces:
        cv2.rectangle(frame, (face[0],face[1]),
                  (face[2],face[3]), (10, 255, 0), 2)
    
    
    #########################################################################################
    # Loop over all tracks and draw detection box if confidence is above minimum threshold
    for track in tracks:
        
        if True:

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            
            xmin = int(track.box[0])
            ymin = int(track.box[1])
            xmax = int(track.box[2])
            ymax = int(track.box[3])
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(track.cl)]
            label = '%s: %s%%' % (object_name, str(float(track.score))) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    #Press 'ESC' to quit 
    k=cv2.waitKey(30) &0xff
    if k==27:
        break 

# Clean up
cv2.destroyAllWindows()
webcam2.stop()


