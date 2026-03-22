/**
 * @file motion.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-15
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "motion/motion.h"

Motion::Motion(ros::NodeHandle &nh) : nh_(nh), quad_(nh), omni_(nh) {
  status_.cfg_ = gecko;
  status_.mode_ = omni;
  status_.w_ = normal;
  status_.h_ = standard;
  waistAngle_ << 90, 90, 180, 180, 180;
  ROS_INFO("[Motion] initialized");
}
Motion::~Motion() {}
void Motion::UpdateStatus(CrimsonParam status, Vec5 waistAngle) {
  status_ = status;
  ROS_INFO("[Motion] status update to (cfg: %d, mode: %d, w: %d, h: %d)",
           status_.cfg_, status_.mode_, status_.w_, status_.h_);
  waistAngle_ << waistAngle;
  ROS_INFO("[Motion] waist angle update to (%.2f, %.2f, %.2f, %.2f, %.2f)",
           waistAngle_[0], waistAngle_[1], waistAngle_[2], waistAngle_[3],
           waistAngle_[4]);
  quad_.UpdateStatus(status_, waistAngle_);
  omni_.UpdateStatus(status_);
}
void Motion::Go(float v) {
  int s = 200, h = 400;
  s = v > 0 ? s : -s;
  quad_.Go(s, h);
}
void Motion::Move(float vx, float vy, float omega) {
  omni_.Move(vx, vy, omega);
}
void Motion::Turn(float theta) {
  switch (status_.mode_) {
    case quad: {
      int th = 30, h = 400;
      th = theta > 0 ? -th : th;
      quad_.Turn(th, h);
      break;
    }
    case wheel: {
      break;
    }
  }
}
void Motion::SetPitch(float pitch) {
  switch (status_.mode_) {
    case quad: {
      quad_.UpdateStatus(status_, waistAngle_);
      quad_.Pitch(pitch);
      break;
    }
    case omni: {
      break;
    }
    case wheel: {
      break;
    }
  }
}

void Motion::SetRoll(float roll) {
  switch (status_.mode_) {
    case quad: {
      quad_.UpdateStatus(status_, waistAngle_);
      quad_.Roll(roll);
      break;
    }
    case omni: {
      break;
    }
    case wheel: {
      break;
    }
  }
}