/**
 * @file omni.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "boost/thread.hpp"
#include "lk_msgs/BrdcstState1.h"
#include "lk_msgs/BrdcstState2.h"
#include "lk_msgs/BrdcstVel.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/time_synchronizer.h"
#include "motion/leg.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "teb_local_planner/FeedbackMsg.h"
#include "teb_local_planner/TrajectoryPointMsg.h"
#include "tf/tf.h"
#include "utils/dxl_interface.h"
#include "utils/log_utils.hpp"
#include "utils/typedefine.h"

typedef message_filters::sync_policies::ApproximateTime<
    teb_local_planner::FeedbackMsg, nav_msgs::Odometry>
    syncPolicyPose;

class Omni {
 private:
  Ti32 defaultJoint_;
  VVi defaultTheta_;
  Tf defaultScale_;

  DynamixelInterface dxl_;
  CrimsonParam status_;
  std::vector<uint8_t> packetID_;
  Eigen::Matrix<float, 4, 3> J_;

  float R_;      // 轮径
  float alpha_;  // 辊子偏角
  Vec3 vel_;     // vx, vy, omega
  Vec4 dps_;     // 目标转速
  Vec3 pv_;

  YAML::Node cfg_;
  bool trackerEnable_ = false;
  float J0_;
  float b0_;
  float kd_;
  float k1_;
  float k2_;
  float k3_;
  float p_;
  float q_;
  float death_;
  CVec3 e_;
  ros::Time initStamp_;

  // cmd vel K
  float Kxy_ = 0;
  float Ko_ = 0;
  float xyThres_;
  float omegaThres_;

  ros::NodeHandle nh_;
  ros::AsyncSpinner spinner_;
  ros::Publisher ptPub_;
  ros::Subscriber trackSub_;
  ros::Subscriber enSub_;
  message_filters::Subscriber<teb_local_planner::FeedbackMsg>* refSub_;
  message_filters::Subscriber<nav_msgs::Odometry>* odomSub_;
  message_filters::Synchronizer<syncPolicyPose>* synchronizer_;
  ros::ServiceClient velClient_;

  bool omniEnable_ = false;
  bool trackEnable_ = false;
  bool flagStop_ = true;
  bool flagInit_ = true;

  void ParamInit();
  void ROSInit();
  void UpdateJ(float theta, float W, float H);
  float CalcYaw(const geometry_msgs::Quaternion& msg);
  void Run();
  void BrdcstVel0();

 public:
  explicit Omni(ros::NodeHandle& nh);
  ~Omni();
  void UpdateStatus(CrimsonParam status);
  void Go(float v);
  void Turn(float omega);
  void Move(float vx, float vy, float omega);

  void RefOdomSyncCallback(const teb_local_planner::FeedbackMsgConstPtr& ref,
                           const nav_msgs::Odometry::ConstPtr& odom);
  void TrackCallback(const std_msgs::Bool::ConstPtr& msg);
  void EnCallback(const std_msgs::Bool::ConstPtr& msg);
};
