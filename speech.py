import sys
import subprocess
from gtts import gTTS
import os
import time

#text = "Lucas is on the right"


def speak(temp):
    voice = gTTS(text=temp, lang="en")
    voice.save("/Users/suhyeonyoo/Desktop/face_recog/temp.mp3")
    opener ="open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, "/Users/suhyeonyoo/Desktop/face_recog/temp.mp3"])
    print("Speaking.....")
    time.sleep(1)
    os.remove("/Users/suhyeonyoo/Desktop/face_recog/temp.mp3")

#speak(text)