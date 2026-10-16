/**
 * @file typedefine.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __TYPE_DEFINE_H__
#define __TYPE_DEFINE_H__

#include <iostream>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "get_param.hpp"
#include "math.h"
#include "math_utils.hpp"
#include "yaml-cpp/yaml.h"

using namespace std;

typedef Eigen::Matrix4f Mat4;
typedef Eigen::Matrix3f Mat3;
typedef Eigen::Vector4f Vec4;
typedef Eigen::Vector3f Vec3;
typedef Eigen::Vector3cf CVec3;
typedef Eigen::Vector2f Vec2;
typedef Eigen::AngleAxisf Axis;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Quaternionf Quat;
typedef vector<uint8_t> Vuc;
typedef vector<uint32_t> Vui;
typedef vector<float> Vf;
typedef vector<Vf> VVf;
typedef vector<int> Vi;
typedef vector<Vi> VVi;
typedef vector<vector<vector<vector<int> > > > Ti;
typedef vector<vector<vector<vector<float> > > > Tf;
typedef vector<vector<vector<vector<uint32_t> > > > Ti32;

typedef struct {
  float ret;
  Vec2 theta2;
  Vec2 theta3;
  Vec2 bd;
  Vec2 whatever;
} Boundary;

typedef struct {
  int jointNum_;                   // 活动关节数目
  int legNum_;                     // 腿数目
  int h_;                          // 机身主轴线到腿部基准平面高度
  std::vector<float> waistParam_;  // 机身参数
  Vec3 legLength_;                 // 腿部长度
  std::vector<float> legOffset_;   // 腿部角度偏差
} QuadParam;

typedef enum { gecko, spider, stick, dog, packup } Config;
typedef enum { quad, omni, wheel } Mode;
typedef enum { narrow, normal, extend } Width;
typedef enum { up, standard, down } Height;
typedef struct {
  Config cfg_;
  Mode mode_;
  Width w_;
  Height h_;
} CrimsonParam;

/// 角度制换算成弧度制
#define DEG2RAD M_PI / 180
/// 弧度制换算成角度制
#define RAD2DEG 180 / M_PI

#endif  // __TYPE_DEFINE_H__