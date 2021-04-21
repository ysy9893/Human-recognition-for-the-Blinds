# Google Speech Recog API
# Count Limit: 50 Call per a day

import speech_recognition as sr
import pyaudio

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

# microphone에서 auido source를 생성합니다
r = sr.Recognizer()
r.energy_threshold = 300

with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)
    print("Noise corrected")
    audio = r.listen(source)
    print("Time over!")


# 구글 웹 음성 API로 인식하기 (하루에 제한 50회)
try:
    print("Google Speech Recognition thinks you said : " + r.recognize_google(audio, language = 'ko'))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# 결과
# Google Speech Recognition thinks you said : 안녕하세요