/**
 * @file st.hpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-04-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef __ST_H__
#define __ST_H__

#include "ros/ros.h"
#include "serial/serial.h"
#include "std_msgs/Bool.h"
#include "boost/thread.hpp"
#include "get_param.hpp"


class ST
{
private:
  YAML::Node st_;
  serial::Serial sp_;
  std::string port_;
  int baudrate_;
  int timeout_;
  int looprate_;
  ros::NodeHandle nh_;
  ros::AsyncSpinner spinner_;
  ros::Subscriber relaySub_;
public:
  explicit ST(const ros::NodeHandle &nh);
  ~ST();
  void Read();
  void RelayCallback(const std_msgs::BoolConstPtr &msg);
};




#endif // __ST_H__