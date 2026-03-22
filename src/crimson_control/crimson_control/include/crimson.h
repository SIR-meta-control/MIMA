/**
 * @file crimson.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief 自主变胞四足机器人控制
 * @version 0.1
 * @date 2024-02-22
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __CRIMSON_H__
#define __CRIMSON_H__

#include "crimson_msgs/Motion.h"
#include "crimson_msgs/Trans.h"
#include "motion/motion.h"
#include "trans/trans.h"

using namespace std;

class Crimson {
 private:
  boost::thread runThread_;

  CrimsonParam status_;  // 机器人当前状态
  Trans trans_;          // 形态变换句柄
  Motion motion_;        // 运动句柄

  ros::NodeHandle nh_;         // ROS节点句柄
  ros::Subscriber transSub_;   // 状态更新订阅器
  ros::Subscriber motionSub_;  // 运动订阅器
  ros::Subscriber enSub_;      // 自动运行开关订阅器
  ros::Publisher plannerPub_;

  bool autoRunEn_ = false;
  int delayTime_ = 0;
  YAML::Node cfg_;
  VVi transList_;
  VVf motionList_;
  VVi motionGroupList_;
  void Run();

 public:
  /**
   * @brief Construct a new Crimson object
   *
   * @param nh
   */
  explicit Crimson(ros::NodeHandle &nh);
  /**
   * @brief Destroy the Crimson object
   *
   */
  ~Crimson();
  /**
   * @brief
   *
   * @param msg
   */
  void TransCallback(crimson_msgs::TransConstPtr msg);
  /**
   * @brief
   *
   * @param msg
   */
  void MotionCallback(crimson_msgs::MotionConstPtr msg);
  /**
   * @brief
   *
   * @param msg
   */
  void AutoCallback(const std_msgs::Bool::ConstPtr &msg);
};

#endif  // __CRIMSON_H__