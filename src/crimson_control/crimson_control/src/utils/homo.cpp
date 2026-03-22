/**
 * @file homo.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "homo.h"

Mat4 R(float theta, int axis) {
  Mat4 res = Mat4::Identity();
  Axis rot(theta * DEG2RAD, gUnitVec3[axis]);
  res.block<3, 3>(0, 0) = rot.toRotationMatrix();
  return res;
}
void T(Mat4 &T, Vec3 trans) { T.block<3, 1>(0, 3) = trans; }
