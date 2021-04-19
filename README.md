# Human recognition for the Blinds

## Requirements 
- motpy==0.09
- matplotlib==3.4.1
- numpy==1.20.2
- opencv-python==4.5.1.48 
- tflite-runtime==1.14.0
- requests==2.25.1
- packages required for OPENCV

apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

- Tensorflow 
version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

if [ $version == "3.7" ]; then
pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
fi

if [ $version == "3.5" ]; then
pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp35-cp35m-linux_armv7l.whl
fi

###### TFLite model for detecting vehicle is from https://github.com/ecd1012/rpi_road_object_detection
###### Face Detect and Product Detect from KAKAO AI API 
### How to convert TFLITE to EdgeTPU using Edgetpu-compiler
