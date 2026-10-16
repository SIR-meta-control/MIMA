/**
 * @file transform.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-14
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __TRANS_H__
#define __TRANS_H__

#include <algorithm>

#include "boost/thread.hpp"
#include "crimson_msgs/Trans.h"
#include "ros/ros.h"
#include "utils/dxl_interface.h"
#include "utils/typedefine.h"

using namespace std;

class Trans {
 private:
  YAML::Node cfg_;
  int jointNum_;
  int stepTime_;
  Vuc packetID_;
  vector<Vui> stdAngle_;
  vector<Vui> stdHexAngle_;
  vector<Vui> omniGecko2Dog_;
  vector<Vui> omniDog2Packup_;
  vector<Vui> omniSpider2Dog_;
  CrimsonParam status_;
  DynamixelInterface dxl_;
  ros::NodeHandle nh_;

  void Dog2Packup();
  void Packup2Dog();
  void Gecko2Spider();
  void Spider2Gecko();
  void Spider2Stick();
  void Stick2Spider();

  void OmniGecko2Dog(bool rev);
  void OmniDog2Packup(bool rev);
  void OmniSpider2Dog(bool rev);

 public:
  explicit Trans(ros::NodeHandle &nh);
  ~Trans();
  Vec5 GetStdAngle();
  void Transform(CrimsonParam status);
  void SetConfig(Config cfg);
  void SetMode(Mode mode);
  void SetW(Width w);
  void SetH(Height h);
  void PackUp();
};

#endif  // __TRANS_H__