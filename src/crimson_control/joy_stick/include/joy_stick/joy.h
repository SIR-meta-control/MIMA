/**
 * @file joy.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef JOY_H
#define JOY_H

#include <ros/ros.h>

#include "boost/thread.hpp"
#include "crimson_msgs/Motion.h"
#include "crimson_msgs/Trans.h"
#include "dynamixel_msgs/GetPos.h"
#include "dynamixel_msgs/SetParam.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Header.h"
#include "std_msgs/UInt8.h"
#include "utils/get_param.hpp"
#include "utils/key.hpp"

typedef enum { gecko, spider, stick, dog, packup } Config;
typedef enum { quad, omni, wheel } Mode;
typedef enum { narrow, normal, extend } Width;
typedef enum { up, standard, down } Height;
typedef struct {
  Config cfg_;
  Mode mode_;
  Width w_;
  Height h_;
} CrimsonParam;

class Joy {
 private:
  YAML::Node cfg_;
  ros::NodeHandle nh_;
  ros::ServiceClient posClient_;
  ros::Publisher motionPub_;
  ros::Publisher transPub_;
  ros::Publisher headPub_;
  ros::Publisher torquePub_;
  ros::Publisher trackPub_;
  ros::Publisher omniEnPub_;
  ros::Publisher quadEnPub_;
  ros::Publisher autoEnPub_;
  ros::Subscriber indexSub_;
  ros::AsyncSpinner spinner_;
  KeyboardCtrl KBC_ = KeyboardCtrl();
  std::vector<std::vector<int> > transSignal_;
  std::vector<std::vector<int> > legalStatus_;
  CrimsonParam status_;

  int vx_;
  int stride_;
  int theta_;
  int mode_;
  int times_ = 20;

 public:
  explicit Joy(ros::NodeHandle& nh);
  ~Joy();
  void MainLoop();
  void ReachGoalCallback(const std_msgs::UInt8::ConstPtr& msg);
};

#endif  // JOY_H