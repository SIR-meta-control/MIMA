/**
 * @file st.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-04-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "st.hpp"

ST::ST(const ros::NodeHandle &nh):nh_(nh), spinner_(4) {
  SetCfg(nh_, "/stserial/st_yaml_path", st_);
  port_ = st_["port"].as<std::string>();
  baudrate_ = st_["baudrate"].as<int>();
  timeout_ = st_["timeout"].as<int>();
  looprate_ = st_["looprate"].as<int>();
  if (port_ == "UNSET") {
    ROS_ERROR("Serial port not set!!!");
    ros::shutdown();
    return;
  } else
    ROS_INFO("Param loaded");
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
  relaySub_ = nh_.subscribe<std_msgs::Bool>("/st/set_relay", 1, &ST::RelayCallback, this);
  spinner_.start();
  Read();
}

ST::~ST() {sp_.close();}
void ST::Read(){
  while(true){
    std::vector<uint8_t> buffer;
    size_t byteNum = sp_.available();  // 获取缓冲区内的字节数

    if (byteNum != 0) {
      ROS_INFO("%ld bytes in available buffer", byteNum);
      sp_.read(buffer, byteNum);
      ROS_INFO("Serial read buffer:");
      for (auto item : buffer) 
        std::cout << std::hex << (item & 0xff) << ' ';
      
      std::cout << std::endl;
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  }
}
void ST::RelayCallback(const std_msgs::BoolConstPtr &msg) {
  std::string command = msg->data? "1\r\n": "0\r\n";
  sp_.write(command);
}

