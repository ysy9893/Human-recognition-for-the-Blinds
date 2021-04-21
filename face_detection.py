import sys
import requests
import cv2
from threading import Thread
import face_recognition
import pickle
import cv2
import os
import numpy as np

# find path of xml file containing haarcascade file
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())


width=700
height=700

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(700, 700), src=0):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(src)

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

API_URL_face= 'https://dapi.kakao.com/v2/vision/face/detect'
API_URL_product='https://dapi.kakao.com/v2/vision/product/detect'
MYAPP_KEY = '4cdab2c7ed5f6af444ab8a3316bf914e'

#Initialize Multiple Video streams
cap1=VideoStream(resolution=(width,height),src=0).start()

def detect_face(filename):
    headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

    try:
        files = { 'image' : filename}
        resp_face = requests.post(API_URL_face, headers=headers, files=files)
        resp_face.raise_for_status()

        return resp_face.json()
    except Exception as e:
        print(str(e))
        sys.exit(0)

def detect_product(filename):
    headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

    try:
        files = { 'image' : filename}
        resp_product=requests.post(API_URL_product,headers=headers,files=files)
        resp_product.raise_for_status()

        return resp_product.json()
    except Exception as e:
        print(str(e))
        sys.exit(0)


def bbox(frame, detection_result_face,detection_result_product):

    for face,product in zip(detection_result_face['result']['faces'],detection_result_product['result']['objects']):
        x1_f = int(face['x']*width)
        w_f=int(face['w']*width)
        y1_f= int(face['y']*height)
        h_f=int(face['h'] * height)

        x1_p= int(product['x1'] *width)
        y1_p= int(product['y1'] * height)
        x2_p= int(product['x2'] * width)
        y2_p= int(product['y2'] * height)
        cv2.rectangle(frame, (x1_f, y1_f), (x1_f+w_f, y1_f+h_f),
                      (255, 255, 255), 2)
        cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p),
                      (0, 0, 0), 2)

    return frame



while True:

    frame=cap1.read()
    frame=cv2.resize(frame,(width,height))

    ret,file=cv2.imencode('.jpg',frame)
    file=file.tobytes()

    #ret1, frame1 = cv2.VideoCapture(0).read()

    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ret1, frame1 = cv2.VideoCapture(0).read()

    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)
    names = []
    face_locations =[]

    face_found = False

    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values and True for the embeddings it matches closely
        # and False for rest

        distances = distances = face_recognition.face_distance(data["encodings"], encoding)
        min_value = min(distances)

        # set name =unknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match

        if min_value < 0.5:
            index = np.argmin(distances)
            name = data["names"][index]

        # update the list of names
        names.append(name)
        # loop over the recognized faces

        count_l = 0
        count_r = 0

        print(names)

        for i in range(len(names)):
            cv2.putText(frame, names[i], (100, 100*i + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)




    detection_result_face = detect_face(file)
    detection_result_product=detect_product(file)

    frame = bbox(frame, detection_result_face, detection_result_product)
    cv2.imshow('frame',frame)

    # Press 'ESC' to quit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap1.stop()