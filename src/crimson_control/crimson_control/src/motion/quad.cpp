/**
 * @file quad.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "motion/quad.h"

Quad::Quad(ros::NodeHandle &nh) : nh_(nh), dxl_(nh) {
  SetCfg(nh_, "/crimson/quad_yaml_path", cfg_);
  param_.jointNum_ = cfg_["leg_joint_num"].as<int>();
  param_.legNum_ = cfg_["leg_num"].as<int>();
  param_.h_ = cfg_["h"].as<int>();
  startID_ = cfg_["start_id"].as<int>();
  w_ = cfg_["w"].as<int>();
  stepTime_ = cfg_["step_time"].as<int>();
  GetVector(cfg_, "waist_param", param_.waistParam_);
  GetVector(cfg_, "leg_offset", param_.legOffset_);
  vector<float> tmp;
  GetVector(cfg_, "leg_length", tmp);
  param_.legLength_ << tmp[0], tmp[1], tmp[2];
  GetTensor(cfg_, "local_pos", initPl_);
  // load narrow decrease coefficients
  nds_ = cfg_["narrow_decrease"]["stride"].as<float>();
  ndw_ = cfg_["narrow_decrease"]["w"].as<float>();
  ndh_ = cfg_["narrow_decrease"]["h"].as<float>();

  // init packet and leg ids
  for (size_t i = 0; i < param_.legNum_; i++)
    legs_.emplace_back(new Leg(param_.legLength_));
  for (int i = 0; i < param_.jointNum_; i++)
    packetID_.emplace_back(i + startID_);

  enSub_ = nh_.subscribe("/quad/disable", 1, &Quad::EnCallback, this);
  ROS_INFO("[Quad] sub-module initialized");
}
Quad::~Quad() {}
void Quad::UpdateStatus(CrimsonParam status, Vec5 waistAngle) {
  status_.cfg_ = status.cfg_;
  status_.mode_ = status.mode_;
  status_.w_ = status.w_;
  status_.h_ = status.h_;
  quadEnable_ = status_.mode_ == quad ? true : false;
  if (!quadEnable_) return;
  ROS_INFO("[Quad] cfg update to: (%d, %d, %d, %d)", status_.cfg_,
           status_.mode_, status_.w_, status_.h_);
  UpdateTF(waistAngle);
  vector<float> vecf(initPl_[status_.cfg_][status_.w_][status_.h_].begin(),
                     initPl_[status_.cfg_][status_.w_][status_.h_].end());
  Vec4 initPl = Eigen::Map<Vec4>(vecf.data());
  ROS_INFO("[Quad] initPl (%.0f, %.0f, %.0f, %.0f)", initPl[0], initPl[1],
           initPl[2], initPl[3]);
  SetLocalPos2Leg(initPl);
  SavePose();
  Run(false);
}
void Quad::UpdateTF(Vec5 waistAngle) {
  vector<Vec2> coeff = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
  for (size_t i = 0; i < param_.legNum_; i++) {
    Mat4 Y, Z;
    Y = R((waistAngle[0] - 90) * 0.5 * pow(-1, i), 1);
    Z = R(coeff[i].dot(Vec2(90, waistAngle[3] / 2)), 2);
    T(Y, Vec3((param_.waistParam_[1] / 2 +
               param_.waistParam_[0] * cos(waistAngle[3] * DEG2RAD / 2)) *
                  pow(-1, i + 1),
              0, 0));
    T(Z, Vec3(param_.waistParam_[2], -param_.waistParam_[3] * coeff[i][1],
              -1 * param_.h_));

    legs_[i]->SetTF(Y * Z);
    // ROS_INFO("[UpadateTF] Leg %d update to:", i);
    // std::cout << legs_[i]->GetTF() << std::endl;
  }
}
void Quad::SetLocalPos2Leg(vector<int> legID, vector<Vec4> localPos) {
  if (legID.size() != localPos.size()) ROS_WARN("leg localpos num unmatch");
  for (size_t i = 0; i < min(legID.size(), localPos.size()); i++)
    legs_[legID[i]]->SetLocalPos(localPos[i]);
  // SavePose();
}
void Quad::SetLocalPos2Leg(Vec4 localPos) {
  for (size_t i = 0; i < param_.legNum_; i++) {
    int arg = (i == 0 || i == 3) ? 1 : -1;
    Vec4 pos(localPos[0], arg * localPos[1], localPos[2], 1);
    legs_[i]->SetLocalPos(pos);
  }
  // SavePose();
}
void Quad::SetGlobalPos2Leg(vector<int> legID, vector<Vec4> globalPos) {
  if (legID.size() != globalPos.size()) ROS_WARN("leg globalpos num unmatch");
  for (size_t i = 0; i < min(legID.size(), globalPos.size()); i++)
    legs_[legID[i]]->SetGlobalPos(globalPos[i]);
  // SavePose();
}
bool Quad::SavePose() {
  Vf pose, leg;
  for (auto item : legs_) {
    // item->InverseKinematics(cfg_ == dog);
    item->InverseKinematics(status_.h_ != up);
    Vec4 pl = item->GetLocalPos();
    auto joint = item->GetJointAngle();
    for (size_t i = 0; i < joint.size(); i++) leg.emplace_back(joint[i]);
  }
  if (param_.legOffset_.size() != leg.size()) {
    ROS_ERROR("[Save Pose] offset & leg joint unmatch!");
    ros::shutdown();
  }
  int togle = -1;
  for (size_t i = 0; i < leg.size(); i++) {
    if (i == 1 || i == 4 || i == 7 || i == 10)
      togle = 1;
    else
      togle = -1;
    pose.emplace_back(togle * leg[i] + param_.legOffset_[i]);
  }
  poseList_.emplace_back(pose);
  return true;
}
void Quad::WriteMotor(vector<float> pose) {
  Vui data;
  for (auto item : pose) data.emplace_back(uint32_t(item * 2048 / 180));
  dxl_.SetGoalPosition(packetID_, data);
  boost::this_thread::sleep_for(boost::chrono::milliseconds(stepTime_));
}
void Quad::Run(bool inverse) {
  if (poseList_.size() == 0) ROS_ERROR("[Run] Empty pose List!");
  if (inverse) reverse(poseList_.begin(), poseList_.end());
  for (auto ite = poseList_.begin(); ite != poseList_.end(); ite++) {
    if (!quadEnable_) {
      ROS_ERROR("[quad] motion interupt due to mode change");
      break;
    }
    WriteMotor(*ite);
    // int a = 0;
    // while (true) {
    //   cin >> a;
    //   if (a == 1) break;
    // }
  }
  poseList_.clear();
}
void Quad::MoveLeg(int legID, Vec4 motion, bool local, bool internal) {
  if (local)
    legs_[legID]->SetLocalPos(legs_[legID]->GetLocalPos() + motion);
  else
    legs_[legID]->SetGlobalPos(legs_[legID]->GetGlobalPos() + motion);
  if (!internal) SavePose();
}
void Quad::Leg2MulPos(int legID, vector<Vec4> motion, bool local,
                      bool internal) {
  for (auto item : motion) MoveLeg(legID, item, local, internal);
}
void Quad::Legs2Pos(vector<int> legID, Vec4 motion, bool local, bool internal) {
  for (auto item : legID) MoveLeg(item, motion, local, internal);
}
void Quad::MoveLeg(vector<int> legID, vector<Vec4> motion, bool local,
                   bool internal) {
  if (legID.size() != motion.size()) ROS_WARN("leg motion num unmatch");
  for (size_t i = 0; i < min(legID.size(), motion.size()); i++)
    MoveLeg(legID[i], motion[i], local, internal);
}
void Quad::MoveBody(Vec4 motion, bool internal) {
  for (auto item : legs_) item->SetGlobalPos(item->GetGlobalPos() - motion);
  if (!internal) SavePose();
}
void Quad::SwingLeg(int legID, Vec3 param, Vec4 direction) {
  Vec4 motion = 0.5 * param[0] * direction;
  // 上升h/2
  MoveLeg(legID, {Vec4(0, 0, 0.5 * param[2], 0)}, false, false);
  // 上升h/2 + 前进方向移动l/2 + 向外伸展w
  Leg2MulPos(legID, {Vec4(0, 0, 0.5 * param[2], 0), motion}, false, true);
  Vec4 w1 =
      status_.cfg_ == dog ? Vec4(0, 0, -param[1], 0) : Vec4(param[1], 0, 0, 0);
  MoveLeg(legID, w1, true, false);
  // 下降h/2 + 前进方向移动l/2 + 向内收缩w
  MoveLeg(legID, motion, false, true);
  Vec4 w2 =
      status_.cfg_ == dog ? Vec4(0, 0, param[1], 0) : Vec4(-param[1], 0, 0, 0);
  MoveLeg(legID, w2, true, true);
  MoveLeg(legID, {Vec4(0, 0, -0.5 * param[2], 0)}, false, false);
  // 下降h/2
  MoveLeg(legID, {Vec4(0, 0, -0.5 * param[2], 0)}, false, false);
}
void Quad::SwingLeg(vector<int> legID, Vec3 param, Vec4 direction) {
  for (auto item : legID) SwingLeg(item, param, direction);
}
void Quad::RotateLeg(float theta) {
  for (auto item : legs_)
    item->SetGlobalPos(R(-1 * theta, 2) * item->GetGlobalPos());
  SavePose();
}
void Quad::RotateLeg(vector<int> legID, Vec3 param, float theta) {
  for (auto item : legID) {
    Vec4 po = legs_[item]->GetGlobalPos();
    Vec4 motion = R(theta, 2) * po - po;
    SwingLeg(item, param, motion);
  }
}
void Quad::RotateBody(Vec3 RPY) {
  Mat4 X = R(-1 * RPY[1], 0);
  Mat4 Y = R(-1 * RPY[0], 1);
  Mat4 Z = R(-1 * RPY[2], 2);
  for (auto item : legs_) item->SetGlobalPos(Z * X * Y * item->GetGlobalPos());
  SavePose();
}
void Quad::StandUp() {
  vector<float> vecf(initPl_[status_.cfg_][status_.w_][status_.h_].begin(),
                     initPl_[status_.cfg_][status_.w_][status_.h_].end());
  Vec4 initPl = Eigen::Map<Vec4>(vecf.data());
  SetLocalPos2Leg(initPl);
  Run(false);
}
void Quad::Turn(float theta, float h) {
  bool inv = theta > 0 ? true : false;
  if (theta < 0) theta = -theta;
  vector<vector<int> > group;
  Vec3 lwh;
  if (status_.cfg_ == gecko || status_.cfg_ == spider) {
    group = {{0, 2}, {3, 1}};
    lwh << 1, w_, -h;  // 30: 向外延伸的距离，可调参数
  } else if (status_.cfg_ == stick) {
    group = {{0, 1}, {3, 2}};
    lwh << 1, w_, -h;
  } else if (status_.cfg_ == dog) {
    group = {{2, 0}, {1, 3}};
    lwh << 1, w_, -h;
  }
  vector<float> vecf(initPl_[status_.cfg_][status_.w_][status_.h_].begin(),
                     initPl_[status_.cfg_][status_.w_][status_.h_].end());
  Vec4 initPl = Eigen::Map<Vec4>(vecf.data());
  SetLocalPos2Leg(initPl);
  Vec4 po0 = legs_[group[0][0]]->GetGlobalPos();
  Vec4 po1 = legs_[group[0][1]]->GetGlobalPos();
  Mat4 R_ = R(theta * 0.5, 2);
  SetGlobalPos2Leg(group[0], {R_ * po0, R_ * po1});
  SavePose();
  RotateLeg(group[0], lwh, -theta);
  RotateLeg(-0.5 * theta);
  RotateLeg(group[1], lwh, -theta);
  RotateLeg(-0.5 * theta);
  Run(inv);
}
void Quad::Go(float stride, float h) {
  bool inv = stride > 0 ? true : false;
  if (stride < 0) stride = -stride;
  vector<vector<int> > group;
  Vec3 lwh(stride, w_, -h);
  Vec4 dirc(status_.cfg_ == stick ? 1 : 0, status_.cfg_ == stick ? 0 : 1, 0, 0);
  group = {{0, 2}, {1, 3}};
  vector<float> vecf(initPl_[status_.cfg_][status_.w_][status_.h_].begin(),
                     initPl_[status_.cfg_][status_.w_][status_.h_].end());
  Vec4 initPl = Eigen::Map<Vec4>(vecf.data());
  SetLocalPos2Leg(initPl);
  if (status_.w_ == normal || status_.w_ == extend) {
    if (status_.cfg_ == gecko) {
      int x, y, z;
      nh_.param<int>("/quad/gecko_x", x, 10);
      nh_.param<int>("/quad/gecko_y", y, 50);
      nh_.param<int>("/quad/gecko_z", z, 0);
      MoveLeg(0, Vec4(x, y, z, 0), false, true);
      MoveLeg(1, Vec4(-x, 0.5 * y, z, 0), false, true);
      MoveLeg(2, Vec4(x, -0.5 * y, z, 0), false, true);
      MoveLeg(3, Vec4(-x, -y, z, 0), false, true);
      // SavePose();
      Legs2Pos({2, 0}, -0.5 * stride * dirc, false, true);
    } else if (status_.cfg_ == spider) {
      MoveLeg(0, -(stride / 2 + 60) * dirc, false, true);
      MoveLeg(1, -60 * dirc, false, true);
      MoveLeg(3, 60 * dirc, false, true);
      MoveLeg(2, (60 - stride / 2) * dirc, false, true);
    } else if (status_.cfg_ == stick) {
      Legs2Pos({0, 1}, -stride / 2 * dirc, false, true);
      group = {{1, 0}, {3, 2}};
    } else if (status_.cfg_ == dog) {
      for (size_t i = 0; i < param_.legNum_; i++)
        legs_[i]->SetLocalPos(R(i == 0 || i == 3 ? 10 : -10, 2) * initPl);
      Legs2Pos({0, 2}, -stride / 2 * dirc, false, true);
    }
    SavePose();
    SwingLeg(group[0], lwh, dirc);
    MoveBody(stride / 2 * dirc, false);
    SwingLeg(group[1], lwh, dirc);
    MoveBody(stride / 2 * dirc, false);
    Run(inv);
  } else if (status_.w_ == narrow) {
    if (status_.cfg_ == gecko || status_.cfg_ == spider)
      Legs2Pos({2, 0}, -0.5 * stride * nds_ * dirc, false, true);
    else if (status_.cfg_ == stick) {
      Legs2Pos({0, 1}, -0.5 * stride * nds_ * dirc, false, true);
      group = {{1, 0}, {3, 2}};
    }
    SwingLeg(group[0], Vec3(stride * nds_, w_ * ndw_, -h * ndh_), dirc);
    MoveBody(dirc * stride * nds_ * 0.5, false);
    SwingLeg(group[1], Vec3(stride * nds_, w_ * ndw_, -h * ndh_), dirc);
    MoveBody(dirc * stride * nds_ * 0.5, false);
    Run(inv);
  }
}

void Quad::Roll(float theta) {
  Vec3 RPY(status_.cfg_ == stick ? 0 : theta / 2,
           status_.cfg_ == stick ? theta / 2 : 0, 0);
  vector<float> vecf(initPl_[status_.cfg_][status_.w_][status_.h_].begin(),
                     initPl_[status_.cfg_][status_.w_][status_.h_].end());
  Vec4 initPl = Eigen::Map<Vec4>(vecf.data());
  SetLocalPos2Leg(initPl);
  RotateBody(RPY);
  RotateBody(RPY);
}

void Quad::Pitch(float theta) {
  Vec3 RPY(status_.cfg_ == stick ? theta / 2 : 0,
           status_.cfg_ == stick ? 0 : theta / 2, 0);
  vector<float> vecf(initPl_[status_.cfg_][status_.w_][status_.h_].begin(),
                     initPl_[status_.cfg_][status_.w_][status_.h_].end());
  Vec4 initPl = Eigen::Map<Vec4>(vecf.data());
  SetLocalPos2Leg(initPl);
  RotateBody(RPY);
  RotateBody(RPY);
}

void Quad::EnCallback(const std_msgs::Bool::ConstPtr &msg) {
  quadEnable_ = msg->data;
  if (!quadEnable_) ROS_WARN("[Quad] motion disabled");
  if (quadEnable_) ROS_WARN("[Quad] motion enabled");
}
