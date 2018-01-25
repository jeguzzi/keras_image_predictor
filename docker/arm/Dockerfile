FROM jeguzzi/ros:armhf-kinetic-ros-dev
MAINTAINER Jerome Guzzi “jerome@idsia.ch”

RUN apt-get update && apt-get install -y \
   python-pip \
   ros-kinetic-diagnostic-msgs \
   ros-kinetic-diagnostic-updater \
   && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
RUN pip install numpy

RUN apt-get update && apt-get install -y \
   python-matplotlib python-scipy python-skimage python-h5py\
   && rm -rf /var/lib/apt/lists/*

RUN pip install http://ci.tensorflow.org/view/Nightly/job/nightly-pi/lastStableBuild/artifact/output-artifacts/tensorflow-1.5.0rc1-cp27-none-any.whl
RUN pip install keras

COPY . src/keras_image_predictor

RUN catkin build
