/**
 * @file leg.cpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief 基于大然的变胞四足机器人控制代码重构的重载变胞四足机器人腿部运动学
 * @version 0.1
 * @date 2024-02-22
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "motion/leg.h"

Leg::Leg() {}
Leg::Leg(Vec3 length) : length_(length) {}
Leg::Leg(Vec3 length, Mat4 tf) : length_(length), tf_(tf) {
  tfInv_ << tf.inverse();
}
Leg::Leg(Vec3 length, Mat4 tf, Vec4 localPos)
    : length_(length), tf_(tf), localPos_(localPos) {
  tfInv_ << tf.inverse();
  globalPos_ << tf_ * localPos_;
  InverseKinematics(true);
}
Leg::~Leg() {}
void Leg::SetBoundary() {
  // 一般形态 upward = false
  boundary_[0].theta2 << -M_PI_2, M_PI_2;
  boundary_[0].theta3 << -M_PI_2, M_PI_2;
  float theta = M_PI + boundary_[0].theta3[0];
  float r = sqrt(pow(length_[1], 2) + pow(length_[2], 2) -
                 2 * length_[1] * length_[2] * cos(theta));
  boundary_[0].ret =
      acos((pow(length_[1], 2) + pow(r, 2) - pow(length_[2], 2)) / 2 *
           length_[1] * r);
  boundary_[0].bd << -M_PI_2 - boundary_[0].ret, M_PI_2;
  boundary_[0].whatever << -M_PI_2, M_PI_2 - boundary_[0].ret;
  // 小狗形态 upward = true
  boundary_[1].theta2 << 0, M_PI_2;
  boundary_[1].theta3 << -M_PI_2, M_PI_2;
  theta = M_PI - boundary_[1].theta3[1];
  r = sqrt(pow(length_[1], 2) + pow(length_[2], 2) -
           2 * length_[1] * length_[2] * cos(theta));
  boundary_[1].ret =
      acos((pow(length_[1], 2) + pow(r, 2) - pow(length_[2], 2)) / 2 *
           length_[1] * r);
  boundary_[1].bd << -M_PI_2, M_PI_2 + boundary_[1].ret;
  boundary_[1].whatever << M_PI_2, -M_PI_2 + boundary_[1].ret;
}
void Leg::SetLength(Vec3 length) {
  for (size_t i = 0; i < 3; i++) length_[i] = length[i];
}
void Leg::SetTF(Mat4 tf) {
  tf_ << tf;
  tfInv_ << tf_.inverse();
}
Mat4 Leg::GetTF() { return tf_; }
Vec4 Leg::GetLocalPos() { return localPos_; }
void Leg::SetLocalPos(Vec4 localPos) {
  localPos_ << localPos;
  // ROS_WARN("[SetLocalPos] target local pos: (%.2f, %.2f, %.2f, %.2f)",
  //          localPos_[0], localPos_[1], localPos_[2], localPos_[3]);
  // ROS_WARN("[SetLocalPos] tf_");
  // std::cout << tf_ << std::endl;
  globalPos_ << tf_ * localPos_;
  // ROS_WARN("[SetLocalPos] globalPos_: (%.2f, %.2f, %.2f, %.2f)",
  // globalPos_[0],
  //          globalPos_[1], globalPos_[2], globalPos_[3]);
}
Vec4 Leg::GetGlobalPos() { return globalPos_; }
void Leg::SetGlobalPos(Vec4 globalPos) {
  globalPos_ << globalPos;
  localPos_ << tfInv_ * globalPos_;
}
Vec3 Leg::GetJointAngle() { return jointAngle_; }
void Leg::SetJointAngle(Vec3 jointAngle) { jointAngle_ << jointAngle; }
void Leg::ForwardKinematics() {
  Vec3 theta(jointAngle_ * DEG2RAD);
  float L = length_[0] + length_[1] * cos(theta[1]) +
            length_[2] * cos(theta[1] + theta[2]);
  localPos_ << L * cos(theta[0]), L * sin(theta[0]),
      length_[1] * sin(theta[1]) + length_[2] * sin(theta[1] + theta[2]), 1;
  globalPos_ << tf_ * localPos_;
}
Vec4 Leg::ForwardKinematics(Vec3 jointAngle) {
  SetJointAngle(jointAngle);
  ForwardKinematics();
  std::cout << localPos_ << std::endl;
  return localPos_;
}
float Leg::CalculateTheta(float L) { return atan2(localPos_[2], L); }
float Leg::CalculateBoundary(float theta, bool upward, int r) {
  float outerBoundary;
  float interBoundary;
  if (upward == 0) {
    if (theta > boundary_[0].bd[0] && theta < boundary_[0].bd[1]) {
      if (theta > boundary_[0].whatever[1]) {
        float alpha = boundary_[0].theta2[1] - theta;
        float beta = asin(length_[1] * sin(alpha) / length_[2]);
        float gama = M_PI - alpha - beta;
        interBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                2 * length_[1] * length_[2] * cos(gama),
                            0.5);
        outerBoundary = length_[1] + length_[2];
      } else {
        if (theta > boundary_[0].whatever[0]) {
          interBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                  2 * length_[1] * length_[2] *
                                      cos(M_PI + boundary_[0].theta3[0]),
                              0.5);
          outerBoundary = length_[1] + length_[2];
        } else {
          float alpha = boundary_[0].theta2[0] - theta;
          float beta = asin(length_[1] * sin(alpha) / length_[2]);
          float gama = M_PI - alpha - beta;
          outerBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                  2 * length_[1] * length_[2] * cos(gama),
                              0.5);
          interBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                  2 * length_[1] * length_[2] *
                                      cos(M_PI + boundary_[0].theta3[0]),
                              0.5);
        }
      }
    } else {
      ROS_ERROR("Failed to find approximate point in workspace!!");
      return 0;
    }
  } else {
    if (theta > boundary_[1].bd[0] && theta < boundary_[1].bd[1]) {
      if (theta < boundary_[1].whatever[1]) {
        float alpha = theta - boundary_[1].theta2[0];
        float beta = asin(length_[1] * sin(alpha) / length_[2]);
        float gama = M_PI - alpha - beta;
        interBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                2 * length_[1] * length_[2] * cos(gama),
                            0.5);
        outerBoundary = length_[1] + length_[2];
      } else {
        if (theta < boundary_[1].whatever[0]) {
          interBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                  2 * length_[1] * length_[2] *
                                      cos(M_PI - boundary_[1].theta3[1]),
                              0.5);
          outerBoundary = length_[1] + length_[2];
        } else {
          float alpha = theta - boundary_[1].theta2[1];
          float beta = asin(length_[1] * sin(alpha) / length_[2]);
          float gama = M_PI - alpha - beta;
          outerBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                  2 * length_[1] * length_[2] * cos(gama),
                              0.5);
          interBoundary = pow(pow(length_[1], 2) + pow(length_[2], 2) -
                                  2 * length_[1] * length_[2] *
                                      cos(M_PI - boundary_[1].theta3[1]),
                              0.5);
        }
      }
    } else {
      ROS_ERROR("Failed to find approximate point in workspace!!");
      return 0;
    }
  }
  if (r > outerBoundary) {
    return outerBoundary - 0.1;
  } else if (r < interBoundary) {
    return interBoundary + 0.1;
  } else {
    printf("r is out range of (interBoundary, outerBoundary)\n");
    return 0;
  }
}
bool Leg::InverseKinematics(bool upward) {
  float x = localPos_[0];
  float y = localPos_[1];
  float z = localPos_[2];
  float A1 = x * x + y * y;
  float A2 = A1 + z * z;
  float B1 = sqrt(A1);
  float R1 = z;
  float R2 = B1 - length_[0];
  float LL1 = length_[0] * length_[0];
  float LL2 = length_[1] * length_[1];
  float LL3 = length_[2] * length_[2];
  float R3 = (A2 + LL1 + LL2 - LL3 - 2 * length_[0] * B1) / (2 * length_[1]);
  float R4 = (A2 + LL1 + LL3 - LL2 - 2 * length_[0] * B1) / (2 * length_[2]);
  float R12 = R1 * R1 + R2 * R2;
  float DELTA1 = R12 - R3 * R3;
  float DELTA2 = R12 - R4 * R4;
  if (DELTA1 >= 0.0 && DELTA2 >= 0.0) {
    float R23 = R2 + R3;
    float R24 = R2 + R4;
    if (R23 == 0 || R24 == 0) {
      ROS_ERROR("Zero Division Error in inverse kinematics!!");
      return false;
    } else {
      if (upward == 0) {
        jointAngle_[1] = 2 * atan((R1 + sqrt(DELTA1)) / (R23)) * RAD2DEG;
        jointAngle_[2] =
            2 * atan((R1 - sqrt(DELTA2)) / (R24)) * RAD2DEG - jointAngle_[1];
      } else {
        jointAngle_[1] = 2 * atan((R1 - sqrt(DELTA1)) / (R23)) * RAD2DEG;
        jointAngle_[2] =
            2 * atan((R1 + sqrt(DELTA2)) / (R24)) * RAD2DEG - jointAngle_[1];
      }
      jointAngle_[0] = atan2(y, x) * RAD2DEG;
      return true;
    }
  } else {
    if (x < 0) {
      R2 = -B1 - length_[0];
      y = -y;
      x = -x;
    }
    float j1 = atan2(y, x);
    ROS_WARN("Local pos out of range in inverse kinematics!!");
    // printf("pl out of range in leg::inv_kin()!\n");
    float t = CalculateTheta(R2);
    float r = sqrt(R12);
    float app_r = CalculateBoundary(t, upward, r);
    if (app_r > 0) {
      float Len = app_r * cos(t) + length_[0];
      localPos_[0] = cos(j1) * Len;
      localPos_[1] = sin(j1) * Len;
      localPos_[2] = app_r * sin(t);
      ROS_INFO("approximate pl=[%f, %f, %f]\n", localPos_[0], localPos_[1],
               localPos_[2]);
      return InverseKinematics(upward);
    } else {
      return false;
    }
  }
}
Vec3 Leg::InverseKinematics(Vec4 &localPos, bool upward) {
  SetLocalPos(localPos);
  if (InverseKinematics(upward))
    return jointAngle_;
  else {
    ROS_ERROR("Solve inverse kinematics failed!!");
    return {0, 0, 0};
  }
}
