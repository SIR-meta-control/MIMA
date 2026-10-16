/**
 * @file transform.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-14
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "trans/trans.h"

void Trans::Gecko2Spider() {
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[gecko]);
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[spider]);
  ROS_INFO("[Gecko2Spider]");
}
void Trans::Spider2Gecko() {
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[spider]);
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[gecko]);
  ROS_INFO("[Spider2Gecko]");
}
void Trans::Spider2Stick() {
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[spider]);
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[stick]);
  ROS_INFO("[Spider2Stick]");
}
void Trans::Stick2Spider() {
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[stick]);
  dxl_.SetGoalPosition(packetID_, stdHexAngle_[spider]);
  ROS_INFO("[Stick2Spider]");
}
void Trans::OmniGecko2Dog(bool rev) {
  Vuc packet;
  for (size_t i = 0; i < 17; i++) packet.emplace_back(i + 1);
  vector<Vui> temp(omniGecko2Dog_);
  if (rev) {
    reverse(temp.begin(), temp.end());
    ROS_INFO("[Dog2Gecko]");
  } else {
    ROS_INFO("[Gecko2Dog]");
  }
  for (auto item : temp) {
    dxl_.SetGoalPosition(packet, item);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(stepTime_));
  }
}
void Trans::OmniDog2Packup(bool rev) {
  Vuc packet;
  for (size_t i = 0; i < 17; i++) packet.emplace_back(i + 1);
  vector<Vui> temp(omniDog2Packup_);
  if (rev) {
    reverse(temp.begin(), temp.end());
    ROS_INFO("[Packup2Dog]");
  } else
    ROS_INFO("[Dog2Packup]");
  for (auto item : temp) {
    dxl_.SetGoalPosition(packet, item);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(stepTime_));
  }
}
void Trans::OmniSpider2Dog(bool rev) {
  Vuc packet;
  for (size_t i = 0; i < 17; i++) packet.emplace_back(i + 1);
  vector<Vui> temp(omniSpider2Dog_);
  if (rev) {
    reverse(temp.begin(), temp.end());
    ROS_INFO("[Dog2Spider]");
  } else {
    ROS_INFO("[Spider2Dog]");
  }
  for (auto item : temp) {
    dxl_.SetGoalPosition(packet, item);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(stepTime_));
  }
}
Trans::Trans(ros::NodeHandle& nh) : nh_(nh), dxl_(nh) {
  SetCfg(nh_, "/crimson/trans_yaml_path", cfg_);
  jointNum_ = cfg_["waist_joint_num"].as<int>();
  stepTime_ = cfg_["step_time"].as<int>();
  for (int i = 0; i < jointNum_; i++) packetID_.emplace_back(i + 1);
  GetMat<uint32_t>(cfg_, "std_angle", stdAngle_);
  GetMat<uint32_t>(cfg_, "std_angle", stdHexAngle_);
  for (size_t i = 0; i < stdAngle_.size(); i++)
    for (size_t j = 0; j < stdAngle_[i].size(); j++)
      stdHexAngle_[i][j] = 2048 * stdAngle_[i][j] / 180;
  GetMat<uint32_t>(cfg_, "omni_gecko2dog", omniGecko2Dog_);
  GetMat<uint32_t>(cfg_, "omni_dog2packup", omniDog2Packup_);
  status_.cfg_ = gecko;
  status_.mode_ = omni;
  status_.w_ = normal;
  status_.h_ = standard;
  ROS_INFO("[Trans] initialized");
}
Trans::~Trans() {}
Vec5 Trans::GetStdAngle() {
  Vec5 res;
  res << stdAngle_[status_.cfg_][0], stdAngle_[status_.cfg_][1],
      stdAngle_[status_.cfg_][2], stdAngle_[status_.cfg_][3],
      stdAngle_[status_.cfg_][4];
  ROS_INFO("[GetStdAngle] res: (%.2f, %.2f, %.2f, %.2f, %.2f)", res[0], res[1],
           res[2], res[3], res[4]);
  return res;
}
void Trans::Transform(CrimsonParam status) {
  SetConfig(status.cfg_);
  SetMode(status.mode_);
  SetW(status.w_);
  SetH(status.h_);
  status_ = status;
  ROS_INFO("[Trans] status update to (cfg: %d, mode: %d, w: %d, h: %d)",
           status_.cfg_, status_.mode_, status_.w_, status_.h_);
}
void Trans::SetConfig(Config cfg) {
  ROS_INFO("[SetConfig] current status: (cfg: %d, mode: %d, w: %d, h: %d)",
           status_.cfg_, status_.mode_, status_.w_, status_.h_);
  // SetMode(quad);
  // SetW(normal);
  // SetH(standard);
  if (cfg == status_.cfg_) {
    if (cfg != dog && cfg != packup)
      dxl_.SetGoalPosition(packetID_, stdHexAngle_[cfg]);
    ROS_WARN("[SetConfig] cfg is already set to %d, return", cfg);
    return;
  }
  if (status_.cfg_ == gecko) {
    if (cfg == spider) {
      Gecko2Spider();
    } else if (cfg == stick) {
      Gecko2Spider();
      Spider2Stick();
    } else if (cfg == dog) {
      if (status_.mode_ == omni) {
        Gecko2Spider();
        OmniSpider2Dog(false);
      }
    } else if (cfg == packup) {
      if (status_.mode_ == omni) {
        Gecko2Spider();
        OmniSpider2Dog(false);
        OmniDog2Packup(false);
      }
    }
  } else if (status_.cfg_ == spider) {
    if (cfg == gecko) {
      Spider2Gecko();
    } else if (cfg == stick) {
      Spider2Stick();
    } else if (cfg == dog) {
      if (status_.mode_ == omni) OmniSpider2Dog(false);
    } else if (cfg == packup) {
      if (status_.mode_ == omni) {
        OmniSpider2Dog(false);
        OmniDog2Packup(false);
      }
    }
  } else if (status_.cfg_ == stick) {
    if (cfg == gecko) {
      Stick2Spider();
      Spider2Gecko();
    } else if (cfg == spider) {
      Stick2Spider();
    } else if (cfg == dog) {
      Stick2Spider();
      if (status_.mode_ == omni) OmniSpider2Dog(false);
    } else if (cfg == packup) {
      Stick2Spider();
      if (status_.mode_ == omni) {
        OmniSpider2Dog(false);
        OmniDog2Packup(false);
      }
    }
  } else if (status_.cfg_ == dog) {
    if (cfg == packup) {
      if (status_.mode_ == omni) OmniDog2Packup(false);
    } else if (cfg == gecko) {
      if (status_.mode_ == omni) {
        OmniSpider2Dog(true);
        Spider2Gecko();
      }
    } else if (cfg == spider) {
      if (status_.mode_ == omni) OmniSpider2Dog(true);
    } else if (cfg == stick) {
      if (status_.mode_ == omni) OmniSpider2Dog(true);
      Spider2Stick();
    }
  } else if (status_.cfg_ == packup) {
    if (cfg == dog) {
      if (status_.mode_ == omni) OmniDog2Packup(true);
    } else if (cfg == gecko) {
      if (status_.mode_ == omni) {
        OmniDog2Packup(true);
        OmniSpider2Dog(true);
        Spider2Gecko();
      }
    } else if (cfg == spider) {
      if (status_.mode_ == omni) {
        OmniDog2Packup(true);
        OmniSpider2Dog(true);
      }
    } else if (cfg == stick) {
      if (status_.mode_ == omni) {
        OmniDog2Packup(true);
        OmniSpider2Dog(true);
      }
      Spider2Stick();
    }
  }
  status_.cfg_ = cfg;
}
void Trans::SetMode(Mode mode) {
  status_.mode_ = mode;
  ROS_INFO("[SetMode] mode set to: %d", mode);
}
void Trans::SetW(Width w) {
  status_.w_ = w;
  ROS_INFO("[SetW] w set to: %d", w);
}
void Trans::SetH(Height h) {
  status_.h_ = h;
  ROS_INFO("[SetH] h set to: %d", h);
}