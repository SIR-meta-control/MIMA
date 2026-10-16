/**
 * @file dynamixel_control.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-01-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __DYNAMIXEL_CONTROL_H__
#define __DYNAMIXEL_CONTROL_H__

#include <ros/ros.h>
#include <serial/serial.h>

#include <boost/thread.hpp>
#include <string>
#include <thread>

#include "dynamixel_msgs/GetParam.h"
#include "dynamixel_msgs/GetPos.h"
#include "dynamixel_msgs/LogData.h"
#include "dynamixel_msgs/Ping.h"
#include "dynamixel_msgs/Reboot.h"
#include "dynamixel_msgs/SetParam.h"
#include "dynamixel_msgs/State.h"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "get_param.hpp"
#include "std_msgs/Bool.h"

// Protocol version
#define PROTOCOL_VERSION 2.0  // Default Protocol version of DYNAMIXEL X series.

using namespace std;
using namespace dynamixel;

typedef struct {
  int baudrate_;
  char* port_;
} SerialConfig;

enum ParamType {
  TorqueEnable,     // 64,  1, Read & Write
  GoalPosition,     // 116, 4, Read & Write
  PresentCurrent,   // 126, 2, Read Only
  PresentVelocity,  // 128, 4, Read Only
  PresentPosition,  // 132, 4, Read Only
  InputVoltage,     // 144, 2, Read Only
  Temperature       // 146, 1, Read Only
};

class DynamixelControl {
 private:
  YAML::Node dyn_;
  SerialConfig serialCfg_;
  vector<int> address_;     // control table address
  vector<int> paramBytes_;  // control param length
  vector<string> paramStr_;

  PortHandler* portHandler_;
  PacketHandler* packetHandler_;
  vector<GroupSyncRead*> syncRead_;
  vector<GroupSyncWrite*> syncWrite_;
  vector<uint8_t> packetID_;
  uint8_t error_;
  int commRes_;
  double loopRate_;
  // === ros === //
  ros::NodeHandle nh_;
  ros::ServiceServer readSrv_;
  ros::ServiceServer pingSrv_;
  ros::ServiceServer rebootSrv_;
  ros::ServiceServer posSrv_;
  ros::Subscriber writeSub_;
  ros::Subscriber torqueSub_;
  ros::Publisher statePub_;
  ros::Publisher logPub_;

  bool logEnable_;
  bool feedbackEnable_;
  bool startUpTorqueEnable_;
  bool dynamixelEnable_ = false;

  /**
   * @brief ros publisher/subscriber/server/client...
   *
   */
  void RosInit();
  /**
   * @brief get param from config files
   *
   */
  void ParamInit();
  /**
   * @brief initialize packet
   *
   */
  void PacketInit();
  /**
   * @brief
   *
   */
  void PacketScan();
  /**
   * @brief
   *
   * @param type
   * @param ids
   * @param data
   */
  void SyncWrite(uint8_t type, vector<uint8_t> ids, vector<uint32_t> data);
  /**
   * @brief
   *
   * @param type
   * @param ids
   * @param data
   * @return true
   * @return false
   */
  bool SyncRead(uint8_t type, vector<uint8_t> ids, vector<uint32_t>& data);
  bool SyncRead(uint8_t type, vector<uint8_t> ids, vector<uint16_t>& data);
  /**
   * @brief get param from dynamixel
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool SyncGetParamServer(dynamixel_msgs::GetParam::Request& req,
                          dynamixel_msgs::GetParam::Response& res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool PingServer(dynamixel_msgs::Ping::Request& req,
                  dynamixel_msgs::Ping::Response& res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool RebootServer(dynamixel_msgs::Reboot::Request& req,
                    dynamixel_msgs::Reboot::Response& res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool PosServer(dynamixel_msgs::GetPos::Request& req,
                 dynamixel_msgs::GetPos::Response& res);
  /**
   * @brief
   *
   * @param msg
   */
  void SyncSetParamCallback(const dynamixel_msgs::SetParam::ConstPtr& msg);
  /**
   * @brief
   *
   * @param msg
   */
  void TorqueEnableCallback(const std_msgs::BoolConstPtr& msg);

  void Run();

 public:
  /**
   * @brief Construct a new Dynamixel Control object
   *
   * @param nh
   */
  explicit DynamixelControl(const ros::NodeHandle& nh);
  /**
   * @brief Destroy the Dynamixel Control object
   *
   */
  ~DynamixelControl();
};

#endif  //__DYNAMIXEL_CONTROL_H__