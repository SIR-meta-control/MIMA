/**
 * @file node.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-04-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "st.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "st_node");
  ros::NodeHandle nh("~");
  ST st(nh);
  ros::AsyncSpinner spinner(4);
  // spinner.start();
  // ros::waitForShutdown();
  return 0;
}