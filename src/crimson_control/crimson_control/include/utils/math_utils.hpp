/**
 * @file math_utils.hpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-07-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __MATH_UTILS_HPP__
#define __MATH_UTILS_HPP__

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief 限制输入值的范围
 *
 * @param val 输入值
 * @param threshold 限制范围
 * @return float 限制后的值
 */
float Thres(float val, float threshold) {
  if (val > threshold) return threshold;
  if (val < -threshold) return -threshold;
  return val;
}

#endif  // __MATH_UTILS_HPP__
