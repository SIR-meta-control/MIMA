/**
 * @file lk.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "lk.h"

void LK::ParamInit() {
  SetCfg(nh_, "/lk_control/lk_yaml_path", cfg_);
  port_ = cfg_["port"].as<std::string>();
  baudrate_ = cfg_["baudrate"].as<int>();
  timeout_ = cfg_["timeout"].as<int>();
  looprate_ = cfg_["looprate"].as<double>();
  if (port_ == "UNSET") {
    ROS_ERROR("Serial port not set!!!");
    ros::shutdown();
    return;
  } else
    ROS_INFO("Param loaded");
}
void LK::SerialInit() {
  sp_.setPort(port_);
  sp_.setBaudrate(baudrate_);
  serial::Timeout timeout = serial::Timeout::simpleTimeout(timeout_);
  sp_.setTimeout(timeout);
  ROS_INFO("Set serial port: %s, baudrate: %d, timeout: %d", port_.c_str(),
           baudrate_, timeout_);
  try {
    sp_.open();
  } catch (serial::IOException& err) {
    ROS_ERROR("Open serial port %s failed!!", port_.c_str());
    ros::shutdown();
    return;
  }
  ROS_INFO("Check serial port %s ...", port_.c_str());
  if (sp_.isOpen())
    ROS_INFO("Serial %s is opened, entering main loop...", port_.c_str());
  else {
    ROS_ERROR("Serial port is not open!!!");
    ros::shutdown();
    return;
  }
}
void LK::ROSInit() {
  logPub_ = nh_.advertise<lk_msgs::LogUI>("/lk/feedback", 1);
  comSrv_ = nh_.advertiseService("/lk/command", &LK::CommandServer, this);
  velSrv_ = nh_.advertiseService("/lk/cmd_vel", &LK::VelServer, this);
  bVelSrv_ = nh_.advertiseService("/lk/brdcst_vel", &LK::BrdcstVelServer, this);
  s1Srv_ = nh_.advertiseService("/lk/read_state1", &LK::S1Server, this);
  s2Srv_ = nh_.advertiseService("/lk/read_state2", &LK::S2Server, this);
  bS1Srv_ = nh_.advertiseService("/lk/b_read1", &LK::BrdcstS1Server, this);
  bS2Srv_ = nh_.advertiseService("/lk/b_read2", &LK::BrdcstS2Server, this);
}
LK::LK(ros::NodeHandle& nh) : nh_(nh) {
  ParamInit();
  SerialInit();
  ROSInit();
  MainLoop();
}
LK::~LK() { sp_.close(); }
bool LK::CheckSum(std::vector<uint8_t> buffer) {
  if (buffer.size() != 13) return false;
  bool res = true;
  uint64_t sum = 0;
  for (size_t i = 0; i < 4; i++) sum += buffer[i];
  // ROS_INFO("check point 1");
  // std::cout << std::hex << (sum & 0xff) << ' ' << (buffer[4] & 0xff)
  //           << std::endl;
  if ((sum & 0xff) != buffer[4]) res = false;
  sum = 0;
  for (size_t i = 5; i < 12; i++) sum += buffer[i];
  // ROS_INFO("check point 2");
  // std::cout << std::hex << (sum & 0xff) << ' ' << (buffer[12] & 0xff)
  //           << std::endl;
  if ((sum & 0xff) != buffer[12]) res = false;
  // if (!res) ROS_ERROR("[Check Sum] failed!");
  return res;
}
void LK::CheckErr(uint8_t id, uint8_t err) {
  if (err & 0x01) ROS_ERROR("Motor #%d low input voltage!", id);
  if (err & 0x08) ROS_ERROR("Motor #%d over temperature!", id);
}
void LK::Write(std::vector<uint8_t> command) { sp_.write(command); }
void LK::Read(std::vector<uint8_t>& buffer) {
  buffer.clear();
  size_t byteNum = sp_.available();  // 获取缓冲区内的字节数
  if (byteNum != 0) {
    // ROS_INFO("%ld bytes in available buffer", byteNum);
    sp_.read(buffer, byteNum);
    // ROS_INFO("Serial read buffer:");
    // for (auto item : buffer) std::cout << std::hex << (item & 0xff) << ' ';
    // std::cout << std::endl;
  }
}
void LK::State2Callback(uint8_t& id, uint8_t& t, int16_t& i, int16_t& v) {
  std::vector<uint8_t> buffer;
  Read(buffer);
  if (!CheckSum(buffer)) return;
  id = buffer[2];
  t = buffer[5];
  i = Hex2Dec(buffer[6], buffer[7]);
  v = Hex2Dec(buffer[8], buffer[9]);
}
void LK::ReadState1(uint8_t id, uint8_t& t, int16_t& u, int16_t& i,
                    uint8_t& err) {
  Write({0x3e, 0x9a, id, 0x00, uint8_t((0x3e + 0x9a + id) & 0xff)});
  boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
  std::vector<uint8_t> buffer;
  Read(buffer);
  if (!CheckSum(buffer)) return;
  id = buffer[2];
  t = buffer[5];
  u = Hex2Dec(buffer[6], buffer[7]);
  i = Hex2Dec(buffer[8], buffer[9]);
  err = buffer[11];
  CheckErr(id, err);
}
void LK::ReadState2(uint8_t id, uint8_t& t, int16_t& i, int16_t& v) {
  Write({0x3e, 0x9c, id, 0x00, uint8_t((0x3e + 0x9c + id) & 0xff)});
  boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
  State2Callback(id, t, i, v);
}
void LK::CmdVel(uint8_t id, int32_t goalv, uint8_t& t, int16_t& i, int16_t& v) {
  std::vector<uint8_t> command(10);
  command[0] = 0x3e;
  command[1] = 0xa2;
  command[2] = id;
  command[3] = 0x04;
  command[4] = (0x3e + 0xa2 + id + 0x04) & 0xff;
  uint8_t b0, b1, b2, b3;
  Dec2Hex(&b0, &b1, &b2, &b3, goalv);
  command[5] = b0;
  command[6] = b1;
  command[7] = b2;
  command[8] = b3;
  command[9] = (b0 + b1 + b2 + b3) & 0xff;
  Write(command);
  boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
  State2Callback(id, t, i, v);
  boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
}
// === ROS interface === //
bool LK::CommandServer(lk_msgs::Command::Request& req,
                       lk_msgs::Command::Response& res) {
  Write(req.command);
  boost::this_thread::sleep_for(boost::chrono::milliseconds(25));
  Read(res.status);
  return true;
}
bool LK::BrdcstVelServer(lk_msgs::BrdcstVel::Request& req,
                         lk_msgs::BrdcstVel::Response& res) {
  res.t.resize(req.id.size());
  res.i.resize(req.id.size());
  res.v.resize(req.id.size());
  for (size_t i = 0; i < req.id.size(); i++)
    CmdVel(req.id[i], req.v[i], res.t[i], res.i[i], res.v[i]);
  return true;
}
bool LK::VelServer(lk_msgs::CmdVel::Request& req,
                   lk_msgs::CmdVel::Response& res) {
  CmdVel(req.id, req.v, res.t, res.i, res.v);
  return true;
}
bool LK::S1Server(lk_msgs::State1::Request& req,
                  lk_msgs::State1::Response& res) {
  ReadState1(req.id, res.t, res.u, res.i, res.err);
  return true;
}
bool LK::BrdcstS1Server(lk_msgs::BrdcstState1::Request& req,
                        lk_msgs::BrdcstState1::Response& res) {
  res.t.resize(req.id.size());
  res.u.resize(req.id.size());
  res.i.resize(req.id.size());
  res.err.resize(req.id.size());
  for (size_t i = 0; i < req.id.size(); i++)
    ReadState1(req.id[i], res.t[i], res.u[i], res.i[i], res.err[i]);
  return true;
}
bool LK::S2Server(lk_msgs::State2::Request& req,
                  lk_msgs::State2::Response& res) {
  ReadState2(req.id, res.t, res.i, res.v);
  return true;
}
bool LK::BrdcstS2Server(lk_msgs::BrdcstState2::Request& req,
                        lk_msgs::BrdcstState2::Response& res) {
  res.t.resize(req.id.size());
  res.i.resize(req.id.size());
  res.v.resize(req.id.size());
  for (size_t i = 0; i < req.id.size(); i++)
    ReadState2(req.id[i], res.t[i], res.i[i], res.v[i]);
  return true;
}
void LK::MainLoop() {
  ros::Rate loop_rate(looprate_);
  while (ros::ok()) {
    std::vector<int16_t> u(4), curr(4);
    std::vector<unsigned char> t(4), err(4);
    for (uint8_t i = 0; i < 4; i++) ReadState1(i + 1, t[i], u[i], curr[i], err[i]);
    lk_msgs::LogUI log;
    log.header.stamp = ros::Time::now();
    for (auto item : curr) log.i.push_back(item * 0.01);
    for (auto item : u) log.u.push_back(item * 0.01);
    logPub_.publish(log);
    ros::spinOnce();
    loop_rate.sleep();
  }
}