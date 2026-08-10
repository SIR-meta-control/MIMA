cmake_minimum_required(VERSION 3.0.2)
project(ros_mujoco)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  dynamixel_msgs
  message_generation
)

add_service_files(
  FILES
  get_angles.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs dynamixel_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/ros_mujoco_node.py
  scripts/mujoco_experiment_energy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
