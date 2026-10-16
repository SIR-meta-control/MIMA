/**
 * @file omni.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-22
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "motion/omni.h"

Omni::Omni(ros::NodeHandle& nh) : nh_(nh), dxl_(nh), spinner_(4) {
  ParamInit();
  ROSInit();
  ROS_INFO("[Omni] sub-module initialized");
}
Omni::~Omni() {}
void Omni::ParamInit() {
  SetCfg(nh_, "/crimson/omni_yaml_path", cfg_);
  alpha_ = cfg_["alpha"].as<int>();
  R_ = cfg_["R"].as<float>();
  // NTSM
  trackerEnable_ = cfg_["tracker_enable"].as<bool>();
  J0_ = cfg_["J0"].as<float>();
  b0_ = cfg_["b0"].as<float>();
  kd_ = cfg_["kd"].as<float>();
  k1_ = cfg_["k1"].as<float>();
  k2_ = cfg_["k2"].as<float>();
  k3_ = cfg_["k3"].as<float>();
  p_ = cfg_["p"].as<float>();
  q_ = cfg_["q"].as<float>();
  death_ = cfg_["death"].as<float>();
  Kxy_ = cfg_["Kxy"].as<float>();
  Ko_ = cfg_["Ko"].as<float>();
  xyThres_ = cfg_["xy_thres"].as<float>();
  omegaThres_ = cfg_["omega_thres"].as<float>();
  GetMat<int>(cfg_, "default_theta", defaultTheta_);
  GetTensor<uint32_t>(cfg_, "default_joint", defaultJoint_);
  GetTensor<float>(cfg_, "default_scale", defaultScale_);
  for (size_t i = 0; i < 17; i++) packetID_.emplace_back(1 + i);
  vel_ << 0, 0, 0;
  pv_ << 0, 0, 0;
  e_ << 0, 0, 0;
}
void Omni::ROSInit() {
  ptPub_ = nh_.advertise<teb_local_planner::TrajectoryPointMsg>(
      "/omni/track_trajectory", 1);
  enSub_ = nh_.subscribe("/omni/disable", 1, &Omni::EnCallback, this);
  if (trackerEnable_) {
    trackSub_ = nh_.subscribe("/track_enable", 1, &Omni::TrackCallback, this);
    odomSub_ = new message_filters::Subscriber<nav_msgs::Odometry>(
        nh_, "/aft_mapped_to_init", 1);
    refSub_ = new message_filters::Subscriber<teb_local_planner::FeedbackMsg>(
        nh_, "/move_base/TebLocalPlannerROS/teb_feedback", 1);
    synchronizer_ = new message_filters::Synchronizer<syncPolicyPose>(
        syncPolicyPose(10), *refSub_, *odomSub_);
    synchronizer_->registerCallback(
        boost::bind(&Omni::RefOdomSyncCallback, this, _1, _2));
  }
  velClient_ = nh_.serviceClient<lk_msgs::BrdcstVel>("/lk/brdcst_vel");
  spinner_.start();
}
void Omni::UpdateJ(float theta, float W, float H) {
  float phi = (theta + alpha_) * DEG2RAD;
  J_ << sin(phi) / (R_ * cos(alpha_)), -cos(phi) / (R_ * cos(alpha_)),
      (W * sin(phi) + H * cos(phi)) / (2 * R_ * cos(alpha_)),
      sin(phi) / (R_ * cos(alpha_)), cos(phi) / (R_ * cos(alpha_)),
      (-W * sin(phi) - H * cos(phi)) / (2 * R_ * cos(alpha_)),
      sin(phi) / (R_ * cos(alpha_)), cos(phi) / (R_ * cos(alpha_)),
      (W * sin(phi) + H * cos(phi)) / (2 * R_ * cos(alpha_)),
      sin(phi) / (R_ * cos(alpha_)), -cos(phi) / (R_ * cos(alpha_)),
      (-W * sin(phi) - H * cos(phi)) / (2 * R_ * cos(alpha_));
}
float Omni::CalcYaw(const geometry_msgs::Quaternion& msg) {
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg, quat);
  double roll, pitch, yaw;  // 定义存储roll,pitch and yaw的容器
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);  // 进行转换
  return static_cast<float>(yaw);
}
void Omni::BrdcstVel0() {
  lk_msgs::BrdcstVel cmd;
  cmd.request.id = {1, 2, 3, 4};
  cmd.request.v = {0, 0, 0, 0};
  ros::service::waitForService("lk/brdcst_vel");
  if (!velClient_.call(cmd)) ROS_ERROR("Call [lk/brdcst_vel] failed!");
}
void Omni::RefOdomSyncCallback(
    const teb_local_planner::FeedbackMsgConstPtr& ref,
    const nav_msgs::Odometry::ConstPtr& odom) {
  // switch
  if (!omniEnable_ || !trackEnable_) return;
  // calculate current pose through pos & quaternion
  float yaw = CalcYaw(odom->pose.pose.orientation);
  CVec3 Q;
  Q.real() << odom->pose.pose.position.x -
                  defaultScale_[status_.cfg_][status_.w_][status_.h_][2],
      odom->pose.pose.position.y +
          defaultScale_[status_.cfg_][status_.w_][status_.h_][3],
      yaw;
  Q.imag() << 0, 0, 0;
  // get reference pose
  CVec3 Q_r;
  Q_r.real() << ref->trajectories[0].trajectory[1].pose.position.x,
      ref->trajectories[0].trajectory[1].pose.position.y,
      CalcYaw(ref->trajectories[0].trajectory[1].pose.orientation);
  Q_r.imag() << 0, 0, 0;
  CVec3 ddQ_r;
  ddQ_r.real() << ref->trajectories[0].trajectory[1].velocity.linear.x - pv_[0],
      ref->trajectories[0].trajectory[1].velocity.linear.y - pv_[1],
      ref->trajectories[0].trajectory[1].velocity.angular.z - pv_[2];
  ddQ_r.imag() << 0, 0, 0;
  ddQ_r *= 20;
  pv_ << ref->trajectories[0].trajectory[1].velocity.linear.x,
      ref->trajectories[0].trajectory[1].velocity.linear.y,
      ref->trajectories[0].trajectory[1].velocity.angular.z;

  float dis = sqrt(pow(Q[0].real() - Q_r[0].real(), 2) +
                   pow(Q[1].real() - Q_r[1].real(), 2) +
                   pow(Q[2].real() - Q_r[2].real(), 2));
  if (dis < death_) {
    if (flagStop_) {
      flagStop_ = false;
      BrdcstVel0();
      ROS_INFO("[track] goal arrived");
    }
    return;
  }
  flagStop_ = true;
  cout << "=== NTSM ===" << endl;
  cout << "=== pose ===" << endl;
  cout << "dis: " << dis << endl;
  cout << "Q: (" << Q[0] << ", " << Q[1] << ", " << Q[2] << ")" << endl;
  cout << "Q_r: (" << Q_r[0] << ", " << Q_r[1] << ", " << Q_r[2] << ")" << endl;
  cout << "ddQ_r: (" << ddQ_r[0] << ", " << ddQ_r[1] << ", " << ddQ_r[2] << ")"
       << endl;
  // calculate NTSM input
  // tracking error
  CVec3 e = Q - Q_r;
  CVec3 e_dot = (e - e_) * 20;
  e_ = e;
  cout << "=== tracking error ===" << endl;
  cout << "e: (" << e[0] << ", " << e[1] << ", " << e[2] << ")" << endl;
  cout << "e_dot: (" << e_dot[0] << ", " << e_dot[1] << ", " << e_dot[2] << ")"
       << endl;
  // sliding mode variant
  complex<float> s1 = e[0] + k1_ * pow(e_dot[0], p_ / q_);
  complex<float> s2 = e[1] + k2_ * pow(e_dot[1], p_ / q_);
  complex<float> s3 = e[2] + k3_ * pow(e_dot[2], p_ / q_);
  CVec3 s;
  s.real() << s1.real(), s2.real(), s3.real();
  s.imag() << s1.imag(), s2.imag(), s3.imag();
  cout << "=== sliding mode variant ===" << endl;
  cout << "s: (" << s[0] << ", " << s[1] << ", " << s[2] << ")" << endl;
  // equivalent control input
  float theta = defaultTheta_[status_.cfg_][status_.w_] * DEG2RAD;
  float alpha = alpha_ * DEG2RAD;
  float W = defaultScale_[status_.cfg_][status_.w_][status_.h_][0];
  float H = defaultScale_[status_.cfg_][status_.w_][status_.h_][1];
  Eigen::Matrix<complex<float>, 4, 3> h_psi_inv;
  h_psi_inv.real() << sin(alpha + theta + yaw) / (R_ * cos(alpha)),
      -cos(alpha + theta + yaw) / (R_ * cos(alpha)),
      (H * cos(alpha + theta) + W * sin(alpha + theta)) / (2 * R_ * cos(alpha)),
      sin(alpha + theta - yaw) / (R_ * cos(alpha)),
      cos(alpha + theta - yaw) / (R_ * cos(alpha)),
      -(H * cos(alpha + theta) + W * sin(alpha + theta)) /
          (2 * R_ * cos(alpha)),
      sin(alpha + theta - yaw) / (R_ * cos(alpha)),
      cos(alpha + theta - yaw) / (R_ * cos(alpha)),
      (H * cos(alpha + theta) + W * sin(alpha + theta)) / (2 * R_ * cos(alpha)),
      sin(alpha + theta + yaw) / (R_ * cos(alpha)),
      -cos(alpha + theta + yaw) / (R_ * cos(alpha)),
      -(H * cos(alpha + theta) + W * sin(alpha + theta)) /
          (2 * R_ * cos(alpha));
  h_psi_inv.imag() << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  Eigen::Matrix3cf M_inv;
  complex<float> m1 = static_cast<complex<float> >(q_ / (p_ * k1_)) *
                      static_cast<complex<float> >(pow(e_dot[0], 1 - p_ / q_));
  complex<float> m2 = static_cast<complex<float> >(q_ / (p_ * k2_)) *
                      static_cast<complex<float> >(pow(e_dot[1], 1 - p_ / q_));
  complex<float> m3 = static_cast<complex<float> >(q_ / (p_ * k3_)) *
                      static_cast<complex<float> >(pow(e_dot[2], 1 - p_ / q_));
  M_inv.real() << m1.real(), 0, 0, 0, m1.real(), 0, 0, 0, m1.real();
  M_inv.imag() << m1.imag(), 0, 0, 0, m1.imag(), 0, 0, 0, m1.imag();
  Eigen::Vector4cf u_eq = J0_ * h_psi_inv * (ddQ_r - M_inv * e_dot);
  cout << "=== equivalent control input ===" << endl;
  LogMat("h_psi_inv", h_psi_inv);
  LogMat("M_inv", M_inv);
  LogVec("u_eq", u_eq);
  // reaching control input
  Eigen::Matrix3cf B;
  B.real() << 8 * kd_, 0, 0, 0, 8 * kd_, 0, 0, 0, 4 * kd_ / (W + H);
  B.imag() << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  CVec3 sign_s;
  complex<float> ss1 = s[0] / hypot(s[0].real(), s[0].imag());
  complex<float> ss2 = s[1] / hypot(s[1].real(), s[1].imag());
  complex<float> ss3 = s[2] / hypot(s[2].real(), s[2].imag());
  sign_s.real() << ss1.real(), ss2.real(), ss3.real();
  sign_s.imag() << ss1.imag(), ss2.imag(), ss3.imag();
  Eigen::Vector4cf u_r = -h_psi_inv * B * sign_s;
  cout << "=== reaching control input ===" << endl;
  cout << "B:" << endl;
  cout << B(0, 0) << ", " << B(0, 1) << ", " << B(0, 2) << endl;
  cout << B(1, 0) << ", " << B(1, 1) << ", " << B(1, 2) << endl;
  cout << B(2, 0) << ", " << B(2, 1) << ", " << B(2, 2) << endl;
  cout << "u_r: (" << u_r[0] << ", " << u_r[1] << ", " << u_r[2] << ", "
       << u_r[3] << ")" << endl;
  // control input u = u_eq + u_r
  Eigen::Vector4cf u = u_eq + u_r;
  cout << "=== control input ===" << endl;
  cout << "u: (" << u[0] << ", " << u[1] << ", " << u[2] << ", " << u[3] << ")"
       << endl;
  // update goal Q double dot
  float a0 = (R_ * cos(alpha) * cos(yaw)) / (4 * sin(alpha + theta));
  float a1 = (R_ * cos(alpha) * sin(yaw)) / (4 * cos(alpha + theta));
  float b0 = (R_ * cos(alpha) * sin(yaw)) / (4 * sin(alpha + theta));
  float b1 = (R_ * cos(alpha) * cos(yaw)) / (4 * cos(alpha + theta));
  float c = (R_ * cos(alpha)) /
            (2 * (H * cos(alpha + theta) + W * sin(alpha + theta)));
  Eigen::Matrix<complex<float>, 3, 4> h_psi;
  h_psi.real() << a0 + a1, a0 - a1, a0 - a1, a0 + a1, b0 - b1, b0 + b1, b0 + b1,
      b0 - b1, c, -c, c, -c;
  h_psi.imag() << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  Eigen::Vector3cf ddQ = (1 / J0_) * h_psi * u;
  cout << "=== update goal Q double dot ===" << endl;
  LogMat("h_psi", h_psi);
  LogVec("ddQ", ddQ);
  // get Q dot by forWar euler method, Q dot: vx, vy, omega
  vel_ = vel_ + ddQ.real() * 0.05;
  vel_[0] = Thres(vel_[0], xyThres_);
  vel_[1] = Thres(vel_[1], xyThres_);
  vel_[2] = Thres(vel_[2], omegaThres_);
  LogVec("vel", vel_);
  Move(vel_[0], vel_[1], vel_[3]);
  // publish pos, vel & acc
  if (flagInit_) {
    initStamp_ = odom->header.stamp;
    flagInit_ = false;
  }
  teb_local_planner::TrajectoryPointMsg pt;
  pt.time_from_start = odom->header.stamp - initStamp_;
  pt.pose = odom->pose.pose;
  pt.pose.position.x = Q[0].real();
  pt.pose.position.y = Q[1].real();
  pt.pose.orientation = tf::createQuaternionMsgFromYaw(Q[2].real());
  pt.velocity.linear.x = vel_[0];
  pt.velocity.linear.y = vel_[1];
  pt.velocity.angular.z = vel_[2];
  pt.acceleration.linear.x = ddQ[0].real();
  pt.acceleration.linear.y = ddQ[1].real();
  pt.acceleration.angular.z = ddQ[2].real();
  ptPub_.publish(pt);
}
void Omni::UpdateStatus(CrimsonParam status) {
  status_ = status;
  omniEnable_ = status_.mode_ == omni;
  if (omniEnable_) {
    Vui data;
    for (auto item : defaultJoint_[status_.cfg_][status_.w_][status_.h_])
      data.emplace_back(item);
    ROS_INFO("angles:");
    for (auto item : data) std::cout << item << ' ';
    std::cout << std::endl;
    dxl_.SetGoalPosition(packetID_, data);
    UpdateJ(defaultTheta_[status_.cfg_][status_.w_],
            defaultScale_[status_.cfg_][status_.w_][status_.h_][0],
            defaultScale_[status_.cfg_][status_.w_][status_.h_][1]);
    ROS_INFO("J:");
    std::cout << J_ << std::endl;
  }
}
void Omni::Go(float v) { Move(v, 0, 0); }
void Omni::Turn(float omega) { Move(0, 0, omega); }
void Omni::Move(float vx, float vy, float omega) {
  if (omniEnable_) {
    if (status_.cfg_ == 2)
      vel_ << -Kxy_ * vy, -Kxy_ * vx, Ko_ * omega;
    else if (status_.cfg_ == 0 && status_.w_ == 0)
      vel_ << -Kxy_ * vx, -Kxy_ * vy, Ko_ * omega;
    else
      vel_ << Kxy_ * vx, -Kxy_ * vy, -Ko_ * omega;
  } else
    vel_ << 0, 0, 0;
  if (!omniEnable_ ||
      (abs(vel_[0]) < 1e-3 && abs(vel_[1]) < 1e-3 && abs(vel_[2]) < 1e-3)) {
    if (flagStop_) {
      flagStop_ = false;
      BrdcstVel0();
    }
    return;
  }
  flagStop_ = true;
  dps_ = RAD2DEG * 1000 * J_ * Vec3(vel_[0], vel_[1], vel_[2]);
  lk_msgs::BrdcstVel cmd;
  cmd.request.id = {1, 2, 3, 4};
  cmd.request.v = {int32_t(dps_[0]), int32_t(-dps_[1]), int32_t(dps_[2]),
                   int32_t(-dps_[3])};

  ros::service::waitForService("lk/brdcst_vel");
  if (!velClient_.call(cmd)) ROS_ERROR("Call [lk/brdcst_vel] failed!");
  ROS_INFO("cmd vel: (%d, %d, %d, %d)", int32_t(dps_[0]), int32_t(-dps_[1]),
           int32_t(dps_[2]), int32_t(-dps_[3]));
}
void Omni::TrackCallback(const std_msgs::Bool::ConstPtr& msg) {
  trackEnable_ = msg->data;
  ROS_INFO("==== Track %s ===", trackEnable_ ? "Enable" : "Disable");
  BrdcstVel0();
}
void Omni::EnCallback(const std_msgs::Bool::ConstPtr& msg) {
  omniEnable_ = msg->data;
  ROS_WARN("[omni] Motion %s", omniEnable_ ? "Enable" : "Disable");
  if (!msg->data) trackEnable_ = false;
  BrdcstVel0();
}
