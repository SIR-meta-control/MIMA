/**
 * @file dynamixel_control_node.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-01-24
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "dynamixel_control.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "serial_ros_interface");
  ros::NodeHandle nh("~");
  DynamixelControl dc(nh);
  ros::waitForShutdown();
  return 0;
}