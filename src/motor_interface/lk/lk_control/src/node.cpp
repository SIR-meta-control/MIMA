/**
 * @file node.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "lk.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "lk_node");
  ros::NodeHandle nh("~");
  LK lk(nh);
  ros::waitForShutdown();
  return 0;
}