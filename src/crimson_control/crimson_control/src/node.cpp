/**
 * @file node.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "crimson.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "crimson_node");
  ros::NodeHandle nh("~");
  Crimson crimson(nh);
  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
