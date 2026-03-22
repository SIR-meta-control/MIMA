/**
 * @file dxl_interface.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __DXL_INTERFACE_H__
#define __DXL_INTERFACE_H__

#include <vector>

#include "dynamixel_msgs/GetParam.h"
#include "dynamixel_msgs/Ping.h"
#include "dynamixel_msgs/SetParam.h"
#include "ros/ros.h"
#include "typedefine.h"

using namespace std;

class DynamixelInterface {
 private:
  ros::NodeHandle nh_;
  ros::Publisher writePub_;
  ros::ServiceClient pingClient_;
  ros::ServiceClient readClient_;

 public:
  explicit DynamixelInterface(ros::NodeHandle &nh);
  ~DynamixelInterface();
  /**
   * @brief
   *
   * @param ids
   */
  void Ping(Vuc &ids);
  /**
   * @brief
   *
   * @param ids
   * @param res
   * @param type
   */
  void SyncRead(Vuc &ids, Vui &res, int type);
  /**
   * @brief
   *
   * @param ids
   * @param data
   * @param type
   */
  void SyncWrite(Vuc &ids, Vui &data, int type);
  /**
   * @brief Get the Torque Enable object
   *
   * @param ids
   * @param res
   */
  void GetTorqueEnable(Vuc &ids, Vui &res);
  /**
   * @brief Set the Torque Enable object
   *
   * @param ids
   * @param goal
   */
  void SetTorqueEnable(Vuc &ids, Vui &goal);
  /**
   * @brief Get the Goal Posiion object
   *
   * @param ids
   * @param res
   */
  void GetGoalPosition(Vuc &ids, Vui &res);
  /**
   * @brief Set the Torque Enable object
   *
   * @param ids
   * @param goal
   */
  void SetGoalPosition(Vuc &ids, Vui &goal);
  /**
   * @brief Get the Present Current object
   *
   * @param ids
   * @param res
   */
  void GetPresentCurrent(Vuc &ids, Vui &res);
  /**
   * @brief Get the Present Velocity object
   *
   * @param ids
   * @param res
   */
  void GetPresentVelocity(Vuc &ids, Vui &res);
  /**
   * @brief Get the Present Position object
   *
   * @param ids
   * @param res
   */
  void GetPresentPosition(Vuc &ids, Vui &res);
  /**
   * @brief Get the Input Voltage object
   *
   * @param ids
   * @param res
   */
  void GetInputVoltage(Vuc &ids, Vui &res);
  /**
   * @brief Get the Temperature object
   *
   * @param ids
   * @param res
   */
  void GetTemperature(Vuc &ids, Vui &res);
};

#endif  // __DXL_INTERFACE_H__