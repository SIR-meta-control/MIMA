/**
 * @file dxl_interface.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "utils/dxl_interface.h"

DynamixelInterface::DynamixelInterface(ros::NodeHandle &nh) : nh_(nh) {
  writePub_ = nh.advertise<dynamixel_msgs::SetParam>(
      "/dynamixel_control/sync_write", 1);
  pingClient_ =
      nh.serviceClient<dynamixel_msgs::Ping>("/dynamixel_control/ping");
  readClient_ =
      nh.serviceClient<dynamixel_msgs::GetParam>("/dynamixel_msgs/sync_read");
}
DynamixelInterface::~DynamixelInterface() {}
void DynamixelInterface::Ping(Vuc &ids) {
  dynamixel_msgs::Ping ping;
  ros::service::waitForService("/dynamixel_control/ping");
  if (!pingClient_.call(ping)) ROS_ERROR("[Ping] call service [ping] failed!");
  ids.clear();
  for (auto item : ping.response.ids) ids.emplace_back(item);
}
void DynamixelInterface::SyncRead(Vuc &ids, Vui &res, int type) {
  dynamixel_msgs::GetParam read;
  read.request.paramType = type;
  read.request.packetID = ids;
  ros::service::waitForService("/dynamixel_control/sync_read");
  if (!readClient_.call(read)) ROS_ERROR("[Sync Read] #%d failed!", type);
  res.clear();
  for (auto item : read.response.params) res.emplace_back(item);
}
void DynamixelInterface::SyncWrite(Vuc &ids, Vui &data, int type) {
  dynamixel_msgs::SetParam write;
  write.paramType = type;
  // ROS_INFO("[SyncWrite] type: %d", write.paramType);
  write.motorID = ids;
  // ROS_INFO("[SyncWrite] write.motorID: ");
  // for (auto item : write.motorID) std::cout << std::hex << (item & 0xff) << '
  // '; std::cout << std::endl;
  write.params = data;
  // ROS_INFO("[SyncWrite] params: ");
  // for (auto item : write.params) std::cout << item << ' ';
  // std::cout << std::endl;
  writePub_.publish(write);
}
void DynamixelInterface::GetTorqueEnable(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 0);
}

void DynamixelInterface::SetTorqueEnable(Vuc &ids, Vui &goal) {
  SyncWrite(ids, goal, 0);
}

void DynamixelInterface::GetGoalPosition(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 1);
}

void DynamixelInterface::SetGoalPosition(Vuc &ids, Vui &goal) {
  // ROS_INFO("[SetGoalPosition]");
  SyncWrite(ids, goal, 1);
}

void DynamixelInterface::GetPresentCurrent(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 2);
}

void DynamixelInterface::GetPresentVelocity(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 3);
}

void DynamixelInterface::GetPresentPosition(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 4);
}

void DynamixelInterface::GetInputVoltage(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 5);
}

void DynamixelInterface::GetTemperature(Vuc &ids, Vui &res) {
  SyncRead(ids, res, 6);
}