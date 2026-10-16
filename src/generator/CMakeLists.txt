cmake_minimum_required(VERSION 3.0.2)
project(generator)

find_package(catkin REQUIRED COMPONENTS
  meta_msgs
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS meta_msgs rospy std_msgs
)

catkin_install_python(PROGRAMS
  node/main.py
  scripts/generator_service.py
  scripts/robot_config_generator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY generation launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
