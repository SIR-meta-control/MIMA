/**
 * @file node.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "joy_stick/joy.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "ckey_node");
  ros::NodeHandle nh("~");
  Joy joy(nh);
  //   ros::AsyncSpinner spinner(4);
  //   spinner.start();
  ros::waitForShutdown();
  return 0;
}
