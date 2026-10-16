/**
 * @file homo.h Homogeneous Transformatin
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "typedefine.h"

std::vector<Vec3> gUnitVec3 = {Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ()};
/**
 * @brief
 *
 * @param theta
 * @param axis
 * @return Mat4
 */
Mat4 R(float theta, int axis);
/**
 * @brief
 *
 * @param T
 * @param trans
 */
void T(Mat4 &T, Vec3 trans);