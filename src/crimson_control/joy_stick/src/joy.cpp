/**
 * @file joy.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "joy_stick/joy.h"

Joy::Joy(ros::NodeHandle& nh) : nh_(nh), spinner_(4) {
  SetCfg(nh_, "/joy_stick/joy_yaml_path", cfg_);
  vx_ = cfg_["vx"].as<int>();
  stride_ = cfg_["stride"].as<int>();
  theta_ = cfg_["theta"].as<int>();
  mode_ = cfg_["mode"].as<int>();
  GetMat(cfg_, "legal_status", legalStatus_);
  GetMat(cfg_, "trans_signal", transSignal_);
  motionPub_ = nh_.advertise<crimson_msgs::Motion>("/crimson/motion", 1);
  transPub_ = nh_.advertise<crimson_msgs::Trans>("/crimson/transform", 1);
  headPub_ = nh_.advertise<dynamixel_msgs::SetParam>(
      "/dynamixel_control/sync_write", 1);
  torquePub_ =
      nh_.advertise<std_msgs::Bool>("/dynamixel_control/torque_enable", 1);
  omniEnPub_ = nh_.advertise<std_msgs::Bool>("/omni/disable", 1);
  quadEnPub_ = nh_.advertise<std_msgs::Bool>("/quad/disable", 1);
  autoEnPub_ = nh_.advertise<std_msgs::Bool>("/crimson/autorun", 1);
  indexSub_ = nh_.subscribe<std_msgs::UInt8>("/move_base/reach_goal", 1,
                                             &Joy::ReachGoalCallback, this);
  posClient_ =
      nh_.serviceClient<dynamixel_msgs::GetPos>("/dynamixel_control/pos");
  trackPub_ = nh_.advertise<std_msgs::Bool>("/track_enable", 1);
  spinner_.start();
  status_.cfg_ = gecko;
  status_.mode_ = omni;
  status_.w_ = normal;
  status_.h_ = standard;
  MainLoop();
}
Joy::~Joy() {}
void Joy::MainLoop() {
  ros::Rate loop_rate(10);
  while (ros::ok()) {
    auto ch = KBC_.get_keyboard_press_key();
    ROS_INFO("get keyboard press 0x%02X \n", ch);
    if (ch != 0) {
      if (ch == KEY_TAB) {  // 变形
        ROS_INFO("[Joy] enter param-trans setting state ");
        int i = 0;
        int code_cfg, code_mode, code_w, code_h;
        while (i != 4) {
          auto code = KBC_.get_keyboard_press_key();
          if (code == KEY_TAB)
            break;
          else if (code == KEY_0 || code == KEY_1 || code == KEY_2 ||
                   code == KEY_3 || code == KEY_4) {
            if (i == 0) {
              code_cfg = code - KEY_0;
              ROS_INFO("[Joy] set cfg to %d ", code_cfg);
            } else if (i == 1) {
              code_mode = code - KEY_0;
              ROS_INFO("[Joy] set mode to %d ", code_mode);
            } else if (i == 2) {
              code_w = code - KEY_0;
              ROS_INFO("[Joy] set width to %d ", code_w);
            } else if (i == 3) {
              code_h = code - KEY_0;
              ROS_INFO("[Joy] set height to %d ", code_h);
            }
            i++;
          } else
            ROS_INFO("[Joy] illegal input, input again ");
        }
        if (i == 4) {
          size_t j = 0;
          for (j = 0; j < legalStatus_.size(); j++)
            if (legalStatus_[j][0] == code_cfg &&
                legalStatus_[j][1] == code_mode &&
                legalStatus_[j][2] == code_w && legalStatus_[j][3] == code_h) {
              crimson_msgs::Trans trans;
              trans.header.stamp = ros::Time::now();
              status_.cfg_ = (Config)code_cfg;
              status_.mode_ = (Mode)code_mode;
              status_.w_ = (Width)code_w;
              status_.h_ = (Height)code_h;
              trans.cfg = status_.cfg_;
              trans.mode = status_.mode_;
              trans.w = status_.w_;
              trans.h = status_.h_;
              transPub_.publish(trans);
              ROS_INFO("[Joy] send msg /crimson/tranform [%d,%d,%d,%d]",
                       status_.cfg_, status_.mode_, status_.w_, status_.h_);
              break;
            }
          if (j == legalStatus_.size())
            ROS_WARN("[Joy] illegal status, please input again");
        } else
          ROS_INFO("[Joy] exit from param-trans setting state ");
      } else if (ch == KEY_SPACE) {  // 停止
        crimson_msgs::Motion motion;
        motion.vx = 0;
        motion.vy = 0;
        motion.omega = 0;
        motion.theta = 0;
        motionPub_.publish(motion);
        std_msgs::Bool dis;
        dis.data = false;
        autoEnPub_.publish(dis);
        if (status_.mode_ == quad)
          quadEnPub_.publish(dis);
        else if (status_.mode_ == omni)
          omniEnPub_.publish(dis);
      } else if (ch == KEY_ENTER) {
        std_msgs::Bool en;
        en.data = true;
        if (status_.mode_ == quad)
          quadEnPub_.publish(en);
        else if (status_.mode_ == omni)
          omniEnPub_.publish(en);
      } else if (ch == KEY_w) {  // 前进
        crimson_msgs::Motion motion;
        motion.vx = 0.01 * vx_ * times_;
        motionPub_.publish(motion);
      } else if (ch == KEY_s) {  // 后退
        crimson_msgs::Motion motion;
        motion.vx = -0.01 * vx_ * times_;
        motionPub_.publish(motion);
      } else if (ch == KEY_a) {  // 左移
        crimson_msgs::Motion motion;
        motion.vy = 0.01 * vx_ * times_;
        motionPub_.publish(motion);
      } else if (ch == KEY_d) {  // 右移
        crimson_msgs::Motion motion;
        motion.vy = -0.01 * vx_ * times_;
        motionPub_.publish(motion);
      } else if (ch == KEY_q) {  // 左转
        crimson_msgs::Motion motion;
        motion.omega = vx_ * 0.01 * times_;
        motionPub_.publish(motion);
      } else if (ch == KEY_e) {  // 右转
        crimson_msgs::Motion motion;
        motion.omega = -vx_ * 0.01 * times_;
        motionPub_.publish(motion);
      } else if (ch == KEY_PLUS) {  // 加速
        ++times_;
        ROS_INFO("times_ = %d", times_);
      } else if (ch == KEY_MINUS) {  // 减速
        --times_;
        times_ = times_ > 0 ? times_ : 0;
        ROS_INFO("times_ = %d", times_);
      } else if (ch == KEY_u) {  // 小狗抬头
        ros::service::waitForService("/dynamixel_control/pos");
        dynamixel_msgs::GetPos pos;
        if (!posClient_.call(pos))
          ROS_ERROR("Call [/dynamixel_control/pos] failed!");
        dynamixel_msgs::SetParam param;
        param.motorID = {3};
        param.params = {pos.response.pos[2] + 100};
        param.paramType = 1;
        headPub_.publish(param);
      } else if (ch == KEY_i) {  // 小狗低头
        ros::service::waitForService("/dynamixel_control/pos");
        dynamixel_msgs::GetPos pos;
        if (!posClient_.call(pos))
          ROS_ERROR("Call [/dynamixel_control/pos] failed!");
        dynamixel_msgs::SetParam param;
        param.motorID = {3};
        param.params = {pos.response.pos[2] - 100};
        param.paramType = 1;
        headPub_.publish(param);
      } else if (ch == KEY_ESC) {  // 关力矩
        std_msgs::Bool param;
        param.data = false;
        torquePub_.publish(param);
      } else if (ch == KEY_WAVE) {  // 开力矩
        std_msgs::Bool param;
        param.data = true;
        torquePub_.publish(param);
      } else if (ch == KEY_o) {  // 开启跟踪
        std_msgs::Bool param;
        param.data = true;
        trackPub_.publish(param);
      } else if (ch == KEY_p) {  // 关闭跟踪
        std_msgs::Bool param;
        param.data = false;
        trackPub_.publish(param);
      } else if (ch == KEY_BACKSLASH) {
        std_msgs::Bool en;
        en.data = true;
        autoEnPub_.publish(en);
      } else if (ch == KEY_c) {  // dog climb
        ROS_INFO("[Joy] enter climb procedure ");
      }
    }
    loop_rate.sleep();
  }
}

void Joy::ReachGoalCallback(const std_msgs::UInt8::ConstPtr& msg) {
  ROS_INFO("[Joy] recieved reach goal %d", msg->data);
  status_.cfg_ = static_cast<Config>(transSignal_[msg->data][0]);
  status_.mode_ = static_cast<Mode>(transSignal_[msg->data][1]);
  status_.w_ = static_cast<Width>(transSignal_[msg->data][2]);
  status_.h_ = static_cast<Height>(transSignal_[msg->data][3]);
  if (msg->data == 6 || msg->data == 14)
    boost::this_thread::sleep(boost::posix_time::milliseconds(2000));
  crimson_msgs::Trans trans;
  trans.header.stamp = ros::Time::now();
  trans.cfg = transSignal_[msg->data][0];
  trans.mode = transSignal_[msg->data][1];
  trans.w = transSignal_[msg->data][2];
  trans.h = transSignal_[msg->data][3];
  transPub_.publish(trans);
}
