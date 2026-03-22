/**
 * @file log_utils.hpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-07-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __LOG_UTILS_HPP__
#define __LOG_UTILS_HPP__

#include "ros/ros.h"
#include "typedefine.h"

using namespace std;

template <typename T>
void LogMat(string name, Eigen::MatrixBase<T>& mat) {
  ROS_INFO_STREAM(name << ":\n" << mat);
}
template <typename T>
void LogVec(string name, Eigen::MatrixBase<T>& vector) {
  ROS_INFO_STREAM(name << ": ");
  for (int i = 0; i < vector.size(); ++i) {
    cout << vector[i];
    if (i < vector.size() - 1) cout << ", ";
  }
  cout << endl;
}

template <typename T>
void Log(string name, vector<T> data) {
  ROS_INFO_STREAM(name << ": ");
  for (int i = 0; i < data.size(); ++i) {
    cout << data[i];
    if (i < data.size() - 1) cout << ", ";
  }
  cout << endl;
}
template <typename T>
void Log(string name, vector<vector<T>> data) {
  ROS_INFO_STREAM(name << ": ");
  for (int i = 0; i < data.size(); ++i) {
    cout << "[";
    for (int j = 0; j < data[i].size(); ++j) {
      cout << data[i][j];
      if (j < data[i].size() - 1) cout << ", ";
    }
    cout << "]";
    if (i < data.size() - 1) cout << ", ";
  }
  cout << endl;
}
template <typename T>
void Log(string name, vector<vector<vector<T>>> data) {
  ROS_INFO_STREAM(name << ": ");
  for (int i = 0; i < data.size(); ++i) {
    cout << "[";
    for (int j = 0; j < data[i].size(); ++j) {
      cout << "[";
      for (int k = 0; k < data[i][j].size(); ++k) {
        cout << data[i][j][k];
        if (k < data[i][j].size() - 1) cout << ", ";
      }
      cout << "]";
      if (j < data[i].size() - 1) cout << ", ";
    }
    cout << "]";
    if (i < data.size() - 1) cout << ", ";
  }
  cout << endl;
}
template <typename T>
void Log(string name, vector<vector<vector<vector<T>>>> data) {
  ROS_INFO_STREAM(name << ": ");
  for (int i = 0; i < data.size(); ++i) {
    cout << "[";
    for (int j = 0; j < data[i].size(); ++j) {
      cout << "[";
      for (int k = 0; k < data[i][j].size(); ++k) {
        cout << "[";
        for (int l = 0; l < data[i][j][k].size(); ++l) {
          cout << data[i][j][k][l];
          if (l < data[i][j][k].size() - 1) cout << ", ";
        }
        cout << "]";
        if (k < data[i][j].size() - 1) cout << ", ";
      }
      cout << "]";
      if (j < data[i].size() - 1) cout << ", ";
    }
    cout << "]";
    if (i < data.size() - 1) cout << ", ";
  }
  cout << endl;
}

#endif  // __LOG_UTILS_HPP__