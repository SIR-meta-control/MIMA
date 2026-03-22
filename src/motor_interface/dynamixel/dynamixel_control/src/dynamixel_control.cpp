/**
 * @file dynamixel_control.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.2
 * @date 2024-01-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "dynamixel_control.h"

DynamixelControl::DynamixelControl(const ros::NodeHandle& nh) : nh_(nh) {
  ROS_INFO("[Dynamixel Control] Initializing...");
  ParamInit();
  RosInit();
  PacketInit();
  dynamixelEnable_ = true;
  if (feedbackEnable_) {
    ROS_INFO("[Dynamixel Control] Feedback activated, entering main loop...");
    Run();
  } else {
    ROS_INFO(
        "[Dynamixel Control] Initialization completed, waiting for "
        "callback...");
    ros::spin();
  }
}
DynamixelControl::~DynamixelControl() {
  portHandler_->closePort();
  if (logEnable_) ROS_ERROR("Serial port closed.");
  for (auto item : syncRead_) delete (item);
  for (auto item : syncWrite_) delete (item);
}

void DynamixelControl::ParamInit() {
  SetCfg(nh_, "/dynamixel_control/dyn_yaml_path", dyn_);
  boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  logEnable_ = dyn_["log"].as<bool>();
  feedbackEnable_ = dyn_["feedback"].as<bool>();
  loopRate_ = dyn_["looprate"].as<double>();
  std::string port;
  port = dyn_["port"].as<string>();
  if (port == "UNSET") {
    if (logEnable_) ROS_ERROR("Serial port not set!!!");
    ros::shutdown();
    return;
  }
  serialCfg_.port_ = (char*)malloc(15 * sizeof(char));
  strcpy(serialCfg_.port_, port.c_str());
  if (logEnable_)
    ROS_INFO("[Param Init] Loaded serial port: %s", serialCfg_.port_);
  serialCfg_.baudrate_ = dyn_["baudrate"].as<int>();
  if (logEnable_)
    ROS_INFO("[Param Init] Loaded Baudrate: %d", serialCfg_.baudrate_);
  startUpTorqueEnable_ = dyn_["torque"].as<bool>();
  if (logEnable_)
    ROS_INFO("[Param Init] Loaded initial torque enable: %d",
             startUpTorqueEnable_);
  GetVector(dyn_, "address", address_);

  if (address_.size() != 7) {  // enabled 7 command type
    if (logEnable_) ROS_ERROR("Param address set failed!!!");
    ros::shutdown();
    return;
  }
  if (logEnable_) {
    ROS_INFO("[Param Init] Loaded address:");
    for (auto item : address_) std::cout << item << ' ';
    std::cout << std::endl;
  }
  error_ = 0;
  commRes_ = COMM_TX_FAIL;
  paramBytes_ = {1, 4, 2, 4, 4, 2, 1};
  paramStr_ = {"TorqueEnable",    "GoalPosition",    "PresentCurrent",
               "PresentVelocity", "PresentPosition", "InputVoltage",
               "Temperature"};
  if (logEnable_) ROS_INFO("[Param Init] done");
}
void DynamixelControl::PacketInit() {
  // set serial
  portHandler_ = PortHandler::getPortHandler(serialCfg_.port_);
  if (!portHandler_->openPort()) {
    if (logEnable_) ROS_ERROR("[Serial Init] Open port Failed!!");
    ros::shutdown();
  }
  if (!portHandler_->setBaudRate(serialCfg_.baudrate_)) {
    if (logEnable_) ROS_ERROR("[Serial Init] Set Baudrate Failed!!");
    ros::shutdown();
  }
  if (logEnable_) ROS_INFO("[Serial Init] done");
  packetHandler_ = PacketHandler::getPacketHandler(PROTOCOL_VERSION);
  // set command
  for (size_t i = 0; i < address_.size(); i++) {
    GroupSyncRead* read = new GroupSyncRead(portHandler_, packetHandler_,
                                            address_[i], paramBytes_[i]);
    GroupSyncWrite* write = new GroupSyncWrite(portHandler_, packetHandler_,
                                               address_[i], paramBytes_[i]);
    syncRead_.emplace_back(read);
    syncWrite_.emplace_back(write);
  }
  if (logEnable_) ROS_INFO("[Command Init] done");
  // enable torque
  PacketScan();
  if (startUpTorqueEnable_) {
    vector<uint32_t> data;
    for (size_t i = 0; i < packetID_.size(); i++) data.emplace_back(1);
    SyncWrite(TorqueEnable, packetID_, data);
    SyncRead(TorqueEnable, packetID_, data);
    if (logEnable_) {
      ROS_INFO("[Packet Init] Set TorqueEnable: ");
      for (auto item : data) std::cout << item << ' ';
      std::cout << std::endl;
    }
  }
}
void DynamixelControl::RosInit() {
  readSrv_ = nh_.advertiseService("/dynamixel_control/sync_read",
                                  &DynamixelControl::SyncGetParamServer, this);
  pingSrv_ = nh_.advertiseService("/dynamixel_control/ping",
                                  &DynamixelControl::PingServer, this);
  rebootSrv_ = nh_.advertiseService("/dynamixel_control/reboot",
                                    &DynamixelControl::RebootServer, this);
  posSrv_ = nh_.advertiseService("/dynamixel_control/pos",
                                 &DynamixelControl::PosServer, this);
  writeSub_ = nh_.subscribe("/dynamixel_control/sync_write", 1,
                            &DynamixelControl::SyncSetParamCallback, this);
  torqueSub_ = nh_.subscribe("/dynamixel_control/torque_enable", 1,
                             &DynamixelControl::TorqueEnableCallback, this);
  statePub_ =
      nh_.advertise<dynamixel_msgs::State>("/dynamixel_control/state", 1);
  logPub_ = nh_.advertise<dynamixel_msgs::LogData>("/dynamixel_control/log", 1);
  if (logEnable_) ROS_INFO("[ros Init] done");
}
void DynamixelControl::PacketScan() {
  commRes_ = packetHandler_->broadcastPing(portHandler_, packetID_);
  if (commRes_ != COMM_SUCCESS) {
    if (logEnable_) ROS_ERROR("[Packet Init] Packet not found");
    ros::shutdown();
  }
  if (logEnable_) {
    ROS_INFO("[Packet Init] Scanned packet ID:");
    for (auto item : packetID_) std::cout << std::hex << (item & 0xff) << ' ';
    std::cout << std::endl;
  }
}
void DynamixelControl::SyncWrite(uint8_t type, vector<uint8_t> ids,
                                 vector<uint32_t> data) {
  if (type < 0 || type > address_.size() - 1) {
    if (logEnable_)
      ROS_ERROR("[Sync Write] #%s is not supported", paramStr_[type].c_str());
    return;
  }
  if (type != TorqueEnable && type != GoalPosition) {
    if (logEnable_)
      ROS_ERROR("[Sync Write] #%s is read only", paramStr_[type].c_str());
    return;
  }
  for (size_t i = 0; i < ids.size(); i++) {
    auto ite = find(packetID_.begin(), packetID_.end(), ids[i]);
    if (ite == packetID_.end()) {
      if (logEnable_) ROS_WARN("[Sync Write] Packet %d not found", ids[i]);
      continue;
    }
    uint8_t param[paramBytes_[type]];
    if (type == TorqueEnable)
      param[0] = data[i];
    else if (type == GoalPosition) {
      param[0] = DXL_LOBYTE(DXL_LOWORD(data[i]));
      param[1] = DXL_HIBYTE(DXL_LOWORD(data[i]));
      param[2] = DXL_LOBYTE(DXL_HIWORD(data[i]));
      param[3] = DXL_HIBYTE(DXL_HIWORD(data[i]));
    }
    if (!syncWrite_[type]->addParam(ids[i], param)) {
      if (logEnable_)
        ROS_WARN("[Sync Write] Add packet %d to #%s failed", ids[i],
                 paramStr_[type].c_str());
    }
  }
  commRes_ = syncWrite_[type]->txPacket();
  if (commRes_ != COMM_SUCCESS)
    if (logEnable_)
      ROS_ERROR("[Sync Write] Set #%s failed, res: %d", paramStr_[type].c_str(),
                commRes_);
  syncWrite_[type]->clearParam();
  if (logEnable_)
    ROS_INFO("[Sync Write] Sent command #%s done", paramStr_[type].c_str());
}
bool DynamixelControl::SyncRead(uint8_t type, vector<uint8_t> ids,
                                vector<uint32_t>& data) {
  data.clear();
  for (auto item : ids) {
    auto ite = find(packetID_.begin(), packetID_.end(), item);
    if (ite == packetID_.end()) {
      if (logEnable_) ROS_WARN("[Sync Read] Packet %d not found", item);
      continue;
    }
    if (!syncRead_[type]->addParam(item)) {
      if (logEnable_)
        ROS_ERROR("[Sync Read] Add param %d to #%s failed!", item,
                  paramStr_[type].c_str());
      return false;
    }
  }
  commRes_ = syncRead_[type]->txRxPacket();
  if (commRes_ == COMM_SUCCESS) {
    for (auto item : ids) {
      uint32_t param =
          syncRead_[type]->getData(item, address_[type], paramBytes_[type]);
      data.emplace_back(param);
      if (logEnable_)
        ROS_INFO("[Sync Read] Get param #%s: %x from packet ID: %d",
                 paramStr_[type].c_str(), param, item);
    }
    syncRead_[type]->clearParam();
    return true;
  } else {
    if (logEnable_)
      ROS_ERROR("[Sync Read] Get #%s failed, res: %d", paramStr_[type].c_str(),
                commRes_);
    syncRead_[type]->clearParam();
    return false;
  }
}
bool DynamixelControl::SyncRead(uint8_t type, vector<uint8_t> ids,
                                vector<uint16_t>& data) {
  data.clear();
  for (auto item : ids) {
    auto ite = find(packetID_.begin(), packetID_.end(), item);
    if (ite == packetID_.end()) {
      if (logEnable_) ROS_WARN("[Sync Read] Packet %d not found", item);
      continue;
    }
    if (!syncRead_[type]->addParam(item)) {
      if (logEnable_)
        ROS_ERROR("[Sync Read] Add param %d to #%s failed!", item,
                  paramStr_[type].c_str());
      return false;
    }
  }
  commRes_ = syncRead_[type]->txRxPacket();
  if (commRes_ == COMM_SUCCESS) {
    for (auto item : ids) {
      uint16_t param =
          syncRead_[type]->getData(item, address_[type], paramBytes_[type]);
      data.emplace_back(param);
      if (logEnable_)
        ROS_INFO("[Sync Read] Get param #%s: %x from packet ID: %d",
                 paramStr_[type].c_str(), param, item);
    }
    syncRead_[type]->clearParam();
    return true;
  } else {
    if (logEnable_)
      ROS_ERROR("[Sync Read] Get #%s failed, res: %d", paramStr_[type].c_str(),
                commRes_);
    syncRead_[type]->clearParam();
    return false;
  }
}
bool DynamixelControl::SyncGetParamServer(
    dynamixel_msgs::GetParam::Request& req,
    dynamixel_msgs::GetParam::Response& res) {
  return SyncRead(req.paramType, req.packetID, res.params);
}
bool DynamixelControl::PingServer(dynamixel_msgs::Ping::Request& req,
                                  dynamixel_msgs::Ping::Response& res) {
  if (packetID_.size() == 0) return false;
  res.ids = packetID_;
  return true;
}
void DynamixelControl::SyncSetParamCallback(
    const dynamixel_msgs::SetParam::ConstPtr& msg) {
  SyncWrite(msg->paramType, msg->motorID, msg->params);
}
bool DynamixelControl::RebootServer(dynamixel_msgs::Reboot::Request& req,
                                    dynamixel_msgs::Reboot::Response& res) {
  packetHandler_->reboot(portHandler_, req.id, &res.err);
  if (res.err == 0)
    return true;
  else
    return false;
}

void DynamixelControl::TorqueEnableCallback(const std_msgs::BoolConstPtr& msg) {
  vector<uint32_t> param;
  uint32_t data = msg->data ? 1 : 0;
  for (auto item : packetID_) param.emplace_back(data);
  SyncWrite(TorqueEnable, packetID_, param);
}
bool DynamixelControl::PosServer(dynamixel_msgs::GetPos::Request& req,
                                 dynamixel_msgs::GetPos::Response& res) {
  return SyncRead(4, packetID_, res.pos);
}

void DynamixelControl::Run() {
  ros::Rate rate(loopRate_);
  boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
  while (ros::ok()) {
    vector<uint16_t> I;
    vector<uint32_t> U;
    SyncRead(PresentCurrent, packetID_, I);
    SyncRead(InputVoltage, packetID_, U);
    // publish log data
    dynamixel_msgs::LogData log;
    for (auto item : I)
      log.I.emplace_back(
          static_cast<float>(item > 0x7fff ? item - 0xffff : item) * 2.69 *
          0.001);
    for (auto item : U) log.U.emplace_back(item * 0.1);
    log.header.stamp = ros::Time::now();
    logPub_.publish(log);
    ros::spinOnce();
    rate.sleep();
  }
}
