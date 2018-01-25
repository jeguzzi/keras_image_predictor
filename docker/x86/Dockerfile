FROM jeguzzi/ros:kinetic-ros-dev
MAINTAINER Jerome Guzzi “jerome@idsia.ch”

RUN apt-get update && apt-get install -y \
   python-pip \
   ros-kinetic-diagnostic-msgs \
   ros-kinetic-diagnostic-updater \
   && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
RUN pip install numpy keras scikit-image tensorflow
RUN pip install h5py

COPY . src/keras_image_predictor

RUN catkin build
