/**
 * @file lk.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __LK_H__
#define __LK_H__

#include <math.h>

#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "ros/ros.h"
#include "serial/serial.h"

// msgs
#include "lk_msgs/LogTIV.h"
#include "lk_msgs/LogUI.h"

// srv
#include "lk_msgs/BrdcstState1.h"
#include "lk_msgs/BrdcstState2.h"
#include "lk_msgs/BrdcstVel.h"
#include "lk_msgs/CmdVel.h"
#include "lk_msgs/Command.h"
#include "lk_msgs/State1.h"
#include "lk_msgs/State2.h"

// utils
#include "dec2hex.hpp"
#include "get_param.hpp"

#define DEG2RAD (M_PI / 180.0)

class LK {
 private:
  YAML::Node cfg_;
  ros::NodeHandle nh_;
  ros::Publisher logPub_;
  ros::ServiceServer comSrv_;
  ros::ServiceServer velSrv_;
  ros::ServiceServer bVelSrv_;
  ros::ServiceServer s1Srv_;
  ros::ServiceServer s2Srv_;
  ros::ServiceServer bS1Srv_;
  ros::ServiceServer bS2Srv_;

  serial::Serial sp_;
  std::string port_;
  int baudrate_;
  int timeout_;
  int looprate_;

  /**
   * @brief
   *
   */
  void ParamInit();
  /**
   * @brief
   *
   */
  void SerialInit();
  /**
   * @brief
   *
   */
  void ROSInit();
  /**
   * @brief Main loop
   *
   */
  void MainLoop();
  /**
   * @brief Check if buffer correct
   *
   * @param buffer src
   * @return true
   * @return false
   */
  bool CheckSum(std::vector<uint8_t> buffer);
  /**
   * @brief
   *
   * @param id
   * @param err
   */
  void CheckErr(uint8_t id, uint8_t err);
  /**
   * @brief Motor status returned after cmd_vel
   *
   * @param id
   * @param t temperature
   * @param i current
   * @param v velocity
   */
  void State2Callback(uint8_t &id, uint8_t &t, int16_t &i, int16_t &v);
  /**
   * @brief send command from req directly
   *
   * @param req command full text
   * @param res status full text
   * @return true success
   * @return false failed
   */
  bool CommandServer(lk_msgs::Command::Request &req,
                     lk_msgs::Command::Response &res);
  /**
   * @brief broadcast close loop velocity control for most 4 motors
   *
   * @param req ids & velocity
   * @param res returned status
   * @note do not work on MG4010
   * @return true success
   * @return false failed
   */
  bool BrdcstVelServer(lk_msgs::BrdcstVel::Request &req,
                       lk_msgs::BrdcstVel::Response &res);
  /**
   * @brief close loop velocity control for motor
   *
   * @param req id & velocity
   * @param res returned status
   * @return true success
   * @return false failed
   */
  bool VelServer(lk_msgs::CmdVel::Request &req, lk_msgs::CmdVel::Response &res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool S1Server(lk_msgs::State1::Request &req, lk_msgs::State1::Response &res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool BrdcstS1Server(lk_msgs::BrdcstState1::Request &req,
                      lk_msgs::BrdcstState1::Response &res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool S2Server(lk_msgs::State2::Request &req, lk_msgs::State2::Response &res);
  /**
   * @brief
   *
   * @param req
   * @param res
   * @return true
   * @return false
   */
  bool BrdcstS2Server(lk_msgs::BrdcstState2::Request &req,
                      lk_msgs::BrdcstState2::Response &res);

 public:
  /**
   * @brief Construct a new LK object
   *
   * @param nh
   */
  explicit LK(ros::NodeHandle &nh);
  /**
   * @brief Destroy the LK object
   *
   */
  ~LK();
  /**
   * @brief Send command to serial port
   *
   * @param command src
   */
  void Write(std::vector<uint8_t> command);
  /**
   * @brief Read data from serial port
   *
   * @param buffer dst
   */
  void Read(std::vector<uint8_t> &buffer);
  /**
   * @brief Read state 1
   *
   * @param id
   * @param t temperature
   * @param u voltage
   * @param err error code
   */
  void ReadState1(uint8_t id, uint8_t &t, int16_t &u, int16_t &i,
                  uint8_t &err);
  /**
   * @brief Read state 2
   *
   * @param id
   * @param t temperature
   * @param i current
   * @param v velocity
   */
  void ReadState2(uint8_t id, uint8_t &t, int16_t &i, int16_t &v);
  /**
   * @brief
   *
   * @param id
   * @param goalv
   * @param t
   * @param i
   * @param v
   */
  void CmdVel(uint8_t id, int32_t goalv, uint8_t &t, int16_t &i, int16_t &v);
};

#endif  // __LK_H__
