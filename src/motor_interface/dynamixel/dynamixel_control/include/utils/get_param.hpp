/**
 * @file get_param_vector.h
 * @author 陈祈 (inorichen77@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __GET_PARAM_VECTOR_H__
#define __GET_PARAM_VECTOR_H__

#include "ros/ros.h"
#include "string"
#include "yaml-cpp/yaml.h"

typedef std::map<std::string, int> Str2Int;

Str2Int cfgMap = {
    {"gecko", 0}, {"spider", 1}, {"stick", 2}, {"dog", 3}, {"packup", 4}};
Str2Int wMap = {{"narrow", 0}, {"normal", 1}, {"extend", 2}};
Str2Int hMap = {{"up", 0}, {"standard", 1}, {"down", 2}};

void SetCfg(const ros::NodeHandle& nh, const std::string& pathParam,
            YAML::Node& cfg) {
  std::string path;
  nh.param<std::string>(pathParam, path, "None");
  if (path == "None") {
    ROS_ERROR("[SetCfg] load yaml path failed");
    return;
  }
  cfg = YAML::LoadFile(path);
}
template <typename T>
void GetParamVector(const ros::NodeHandle& nh, const std::string& param,
                    std::vector<T>& vec) {
  if (!nh.getParam(param, vec)) ROS_ERROR("[GetVec] load %s failed!");
}
template <typename T>
void GetVector(const YAML::Node& cfg, const std::string& param,
               std::vector<T>& vec) {
  for (const auto& item : cfg[param]) vec.push_back(item.as<T>());
}
template <typename T>
void GetMat(const YAML::Node& cfg, const std::string& param,
            std::vector<std::vector<T> >& mat) {
  for (const auto& item : cfg[param]) {
    std::vector<T> tmp;
    for (const auto& value : item) tmp.push_back(value.as<T>());
    mat.push_back(tmp);
  }
}
template <typename T>
void GetTensor(const YAML::Node& cfg, const std::string& param,
               std::vector<std::vector<std::vector<std::vector<T> > > >& ser) {
  ser.clear();
  ser.resize(5);
  YAML::Node series = cfg[param];
  for (const auto& config : series) {
    int cfgIndex = cfgMap[config.first.as<std::string>()];
    ser[cfgIndex].resize(3);
    for (const auto& w : config.second) {
      int wIndex = wMap[w.first.as<std::string>()];
      ser[cfgIndex][wIndex].resize(3);
      for (const auto& h : w.second) {
        int hIndex = hMap[h.first.as<std::string>()];
        for (const auto& value : h.second)
          ser[cfgIndex][wIndex][hIndex].push_back(value.as<T>());
      }
    }
  }
  ROS_WARN("[GetTensor] loaded ser:");
  for (const auto& config : ser)
    for (const auto& w : config)
      for (const auto& h : w) {
        for (const auto& value : h) std::cout << value << " ";
        std::cout << std::endl;
      }
}

#endif
