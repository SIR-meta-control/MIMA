/**
 * @file crimson.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-02-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "crimson.h"

Crimson::Crimson(ros::NodeHandle& nh)
    : runThread_(boost::bind(&Crimson::Run, this)),
      nh_(nh),
      trans_(nh),
      motion_(nh) {
  transSub_ =
      nh_.subscribe("/crimson/transform", 1, &Crimson::TransCallback, this);
  motionSub_ =
      nh_.subscribe("/crimson/motion", 1, &Crimson::MotionCallback, this);
  enSub_ = nh_.subscribe("/crimson/autorun", 1, &Crimson::AutoCallback, this);
  plannerPub_ = nh_.advertise<crimson_msgs::Trans>("/crimson/transformed", 1);

  SetCfg(nh_, "/crimson/group_yaml_path", cfg_);
  delayTime_ = cfg_["delay"].as<int>();
  GetMat<float>(cfg_, "motion_list", motionList_);
  GetMat<int>(cfg_, "transform_list", transList_);
  GetMat<int>(cfg_, "motion_group", motionGroupList_);
  ROS_INFO("[Crimson] started, waiting for callback...");
}
Crimson::~Crimson() {}
void Crimson::TransCallback(crimson_msgs::TransConstPtr msg) {
  ROS_INFO("[Crimson] get msg (cfg: %d, mode: %d, w: %d, h: %d)", msg->cfg,
           msg->mode, msg->w, msg->h);
  status_.cfg_ = static_cast<Config>(msg->cfg);
  status_.mode_ = static_cast<Mode>(msg->mode);
  status_.w_ = static_cast<Width>(msg->w);
  status_.h_ = static_cast<Height>(msg->h);
  ROS_INFO("[Crimson] status update to (cfg: %d, mode: %d, w: %d, h: %d)",
           status_.cfg_, status_.mode_, status_.w_, status_.h_);
  trans_.Transform(status_);
  motion_.UpdateStatus(status_, trans_.GetStdAngle());
  boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
  crimson_msgs::Trans pub;
  pub.cfg = status_.cfg_;
  pub.mode = status_.mode_;
  pub.w = status_.w_;
  pub.h = status_.h_;
  plannerPub_.publish(pub);
}
void Crimson::MotionCallback(crimson_msgs::MotionConstPtr msg) {
  if (status_.mode_ == quad) {
    if (msg->vx != 0) motion_.Go(msg->vx);
    if (msg->omega != 0) motion_.Turn(msg->omega);
  } else if (status_.mode_ == omni)
    motion_.Move(msg->vx, msg->vy, msg->omega);
}
void Crimson::AutoCallback(const std_msgs::Bool::ConstPtr& msg) {
  autoRunEn_ = msg->data;
  if (autoRunEn_)
    ROS_INFO("[Crimson] auto run enabled");
  else
    ROS_INFO("[Crimson] auto run disabled");
}
void Crimson::Run() {
  while (ros::ok()) {
    for (size_t i = 0; i < transList_.size(); i++) {
      if (!autoRunEn_) break;
      ROS_WARN("[crimson] autorun step %ld", i);
      // int a = 0;
      // while (true) {
      //   ROS_INFO("[crimson] press 1 & enter to continue");
      //   cin >> a;
      //   if (a == 1) break;
      // }
      status_.cfg_ = static_cast<Config>(transList_[i][0]);
      status_.mode_ = static_cast<Mode>(transList_[i][1]);
      status_.w_ = static_cast<Width>(transList_[i][2]);
      status_.h_ = static_cast<Height>(transList_[i][3]);
      trans_.Transform(status_);
      motion_.UpdateStatus(status_, trans_.GetStdAngle());
      ROS_INFO("[autorun] status update to (cfg: %d, mode: %d, w: %d, h: %d)",
               status_.cfg_, status_.mode_, status_.w_, status_.h_);
      // if (i == 0)
      //   boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
      boost::this_thread::sleep_for(boost::chrono::milliseconds(delayTime_));
      for (auto item : motionGroupList_[i]) {
        // int a = 0;
        // while (true) {
        //   ROS_INFO("[crimson] press 1 & enter to continue");
        //   cin >> a;
        //   if (a == 1) break;
        // }
        float vx, vy, omega;
        vx = motionList_[item][0];
        vy = motionList_[item][1];
        omega = motionList_[item][2];
        ROS_INFO("[autorun] motion command (vx: %.1f, vy: %.1f, omega: %.1f)",
                 vx, vy, omega);
        if (status_.mode_ == quad && status_.cfg_ == gecko) {
          if (vx != 0) {
            motion_.Go(vx);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(2000));
          } else if (omega != 0) {
            motion_.Turn(omega);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(2000));
          }
        // } else if (status_.mode_ == omni) {
        //   motion_.Move(vx, vy, omega);
        //   boost::this_thread::sleep_for(
        //       boost::chrono::milliseconds(delayTime_));
        // }
      }
    }
  }
}
}
