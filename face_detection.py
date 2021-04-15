import sys
import argparse
import requests
import cv2
from threading import Thread
import matplotlib.pyplot as plt
import numpy


width=700
height=700
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

API_URL_face= 'https://dapi.kakao.com/v2/vision/face/detect'
API_URL_product='https://dapi.kakao.com/v2/vision/product/detect'
MYAPP_KEY = '4cdab2c7ed5f6af444ab8a3316bf914e'

#Initialize Multiple Video streams
cap1=VideoStream(resolution=(width,height),framerate=30,src=0).start()

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
    detection_result_face = detect_face(file)
    detection_result_product=detect_product(file)

    frame = bbox(frame, detection_result_face,detection_result_product)
    cv2.imshow('frame',frame)

    # Press 'ESC' to quit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap1.stop()