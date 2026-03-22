/**
 * @file leg.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief 基于大然的变胞四足机器人控制代码重构的重载变胞四足机器人腿部运动学
 * @version 0.1
 * @date 2024-02-22
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __LEG_H__
#define __LEG_H__

#include "ros/ros.h"
#include "utils/typedefine.h"

using namespace std;

/**
 * @brief 三关节腿类
 * 爬行类机器人三关节腿类，三个关节轴线方向依次为
 * Yaw-Roll-Roll（在一般笛卡尔坐标系中：Z-X-X）
 * @note
 * 局部坐标系与第一关节坐标系重合
 * (Z轴竖直向上，X轴为第一、二关节中心连线，并指向第第二关节中心，Y轴通过右手法则确定)
 * 第二、三关节轴线正方向与局部坐标系Y轴反方向相同
 */
class Leg {
 private:
  // 腿局部坐标系(与第一关节坐标系重合)到全局坐标系的齐次变换矩阵
  // (globalPos_ = tf_ * localPos_)
  Mat4 tf_;
  Mat4 tfInv_;       // tf的逆矩阵 (localPos_ = tfInv_ * globalPos_)
  Vec4 globalPos_;   // 足尖点全局坐标
  Vec4 localPos_;    // 足尖点局部坐标
  Vec3 jointAngle_;  // 腿三个关节角度（角度制，进行三角运算前需要先换成弧度制）
  Vec3 length_;      // 腿部三个连杆长度
  Boundary
      boundary_[2];  // 工作空间边界参数，一般形态upward=false,小狗upward=true

  /**
   * @brief 计算足尖点与第二关节中心点连线与水平面的夹角
   *
   * @param L 足尖点与第二关节中心点连线在水平面投影的长度
   * @return float 足尖点与第二关节中心点连线与水平面的夹角,弧度制
   */
  float CalculateTheta(float L);
  /**
   * @brief Set the Boundary object
   *
   */
  void SetBoundary();
  /**
   * @brief
   * 求解第二关节中心点与交点（足尖点和第二关节中心的连线与工作空间边界的交点）之间的距离
   *
   * @param theta 足尖点与第二关节中心连线与水平面的夹角（弧度制）
   * @param upward 用来选择反解构型, false 爬虫构型 true 小狗构型
   * @param r 足尖点与第二关节中心的距离
   * @return float 第二关节中心点与边界交点之间的距离
   *  @retval bdr_outer  足尖点超出工作空间外边界，返回与外边界交点的距离
   *  @retval bdr_inter  足尖点超出工作空间内边界，返回与内边界交点的距离
   *  @retval 0  无法求解处近似点
   */
  float CalculateBoundary(float theta, bool upward, int r);

 public:
  Leg();
  /**
   * @brief Construct a new Leg object
   *
   * @param length 腿部三个连杆长度
   */
  explicit Leg(Vec3 length);
  /**
   * @brief Construct a new Leg object
   *
   * @param length 腿部三个连杆长度
   * @param tf 齐次变换矩阵
   */
  explicit Leg(Vec3 length, Mat4 tf);
  /**
   * @brief Construct a new Leg object
   *
   * @param length 腿部三个连杆长度
   * @param tf 齐次变换矩阵
   * @param localPos 足尖点局部坐标
   */
  explicit Leg(Vec3 length, Mat4 tf, Vec4 localPos);
  /**
   * @brief Destroy the Leg object
   *
   */
  ~Leg();
  /**
   * @brief Set the Length object
   *
   * @param length 腿三段长度
   */
  void SetLength(Vec3 length);
  /**
   * @brief 设置腿局部坐标系(与第一关节坐标系重合)到全局坐标系的变换矩阵
   *
   * @param tf 齐次变换矩阵
   */
  void SetTF(Mat4 tf);
  Mat4 GetTF();
  /**
   * @brief Get the Local Pos object
   *
   * @return Vec4
   */
  Vec4 GetLocalPos();
  /**
   * @brief Set the Local Pos object
   *
   * @param localPos 足尖点局部坐标 4x1列向量
   */
  void SetLocalPos(Vec4 localPos);
  /**
   * @brief Get the Global Pos object
   *
   * @return Vec4
   */
  Vec4 GetGlobalPos();
  /**
   * @brief Set the Global Pos object
   *
   * @param globalPos 足尖点全局坐标 4x1列向量
   */
  void SetGlobalPos(Vec4 globalPos);
  /**
   * @brief Get the Joint Angle object
   *
   * @return Vec3
   */
  Vec3 GetJointAngle();
  /**
   * @brief Set the Joint Angle object
   *
   * @param jointAngle 三个关节角度
   */
  void SetJointAngle(Vec3 jointAngle);
  /**
   * @brief 运动学正解：通过关节角度求解当前足尖点位置（局部坐标）
   *
   */
  void ForwardKinematics();
  /**
   * @brief 运动学正解：通过关节角度求解当前足尖点位置（局部坐标）
   *
   * @param jointAngle 三个关节角度
   * @return Vec4 足尖局部坐标
   */
  Vec4 ForwardKinematics(Vec3 jointAngle);
  /**
   * @brief 运动学逆解：通过当前足尖点位置（局部坐标）逆解三个关节角度
   *
   * @param upward 选择逆解构型
   * 对于同一个足尖位置,有两种腿构型,多解
   * upward = false 爬虫构型,第三关节角度比较小
   * upward = true 小狗构型,第三关节角度比较大
   * @note
   * 逆解函数中加入了边界检查函数，当足尖点超出腿部工作空间后，会自动用最近边界点替换并进行逆解
   * @return true
   * @return false
   */
  bool InverseKinematics(bool upward);
  /**
   * @brief 运动学逆解：通过当前足尖点位置（局部坐标）逆解三个关节角度
   *
   * @param localPos 足尖点局部坐标 4x1列向量
   * @param upward
   * @note
   * 逆解函数中加入了边界检查函数，当足尖点超出腿部工作空间后，会自动用最近边界点替换并进行逆解
   * @return Vec3 三个关节角度
   */
  Vec3 InverseKinematics(Vec4 &localPos, bool upward);
};

#endif  // __LEG_H__
