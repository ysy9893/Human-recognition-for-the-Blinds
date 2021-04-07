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
import time
import importlib.util
import imutils
from imutils.video import VideoStream
from nms import NMS
from motpy import Detection, MultiObjectTracker

#####################################################################################
#####################################################################################
## Define and parse input arguments
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

## Initialize multiple video stream

#webcam1=VideoStream(usePicamera=True).start()
webcam2=VideoStream(src=0).start()
#webcam1=VideoStream(usePicamera=True).start()
##Initialize Tracker 
tracker=MultiObjectTracker(dt=0.1) #100ms

##Get video stream feeds 
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    #Grab frame from video stream
    
    #frame1,frame2= webcam1.read(),webcam2.read()
    frame1=webcam2.read()
    #frame1=cv2.hconcat([frame1,frame2])


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
    
    print(classes)
    ##Tracking 
    detections=[Detection(box=bbox,score=sc,cl=cl) for bbox, sc, cl in zip(boxes,scores,classes)]
    tracker.step(detections)
    tracks=tracker.active_tracks()
    

    ###################################################################################################
    
    # Loop over all tracks and draw detection box if confidence is above minimum threshold
    for track in tracks:
        
        if True:

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            
            xmin = int(track.box[0])
            ymin = int(track.box[1])
            xmax = int(track.box[2])
            ymax = int(track.box[3])
            
            frame=cv2.resize(frame,(imW,imH))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            print(track.cl)
            object_name = labels[int(track.cl)]
            label = '%s: %s%%' % (object_name, str(int(track.score))) # Example: 'person: 72%'
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
imutils.stop()

