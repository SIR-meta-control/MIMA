/**
 * @file motion.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __MOTION_H__
#define __MOTION_H__

#include "motion/omni.h"
#include "motion/quad.h"
#include "utils/typedefine.h"

class Motion {
 public:
  ros::NodeHandle nh_;
  CrimsonParam status_;
  Vec5 waistAngle_;

  Quad quad_;
  Omni omni_;

  explicit Motion(ros::NodeHandle &nh);
  ~Motion();
  void UpdateStatus(CrimsonParam status, Vec5 waistAngle);

  void Go(float v);
  void Move(float vx, float vy, float omega);
  void Turn(float angle);
  void SetPitch(float pitch);
  void SetRoll(float roll);
};

#endif  // __MOTION_H__