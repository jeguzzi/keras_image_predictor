cmake_minimum_required(VERSION 2.8.3)
project(keras_image_predictor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  diagnostic_updater
  diagnostic_msgs
)

catkin_python_setup()

catkin_package(
   CATKIN_DEPENDS std_msgs diagnostic_updater diagnostic_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
