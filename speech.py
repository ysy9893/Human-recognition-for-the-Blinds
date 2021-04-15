import sys
import subprocess
from gtts import gTTS
import os
import time

#text = "Lucas is on the right"


def speak(temp):
    voice = gTTS(text=temp, lang="ko")
    voice.save("./temp.mp3")
    opener ="open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, "./temp.mp3"])
    print("Speaking.....")
    time.sleep(1)
    os.remove("./temp.mp3")

#speak(text)