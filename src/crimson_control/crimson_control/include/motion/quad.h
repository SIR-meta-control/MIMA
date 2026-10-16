/**
 * @file quad.h
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __QUAD_H__
#define __QUAD_H__

#include <std_msgs/Bool.h>

#include <algorithm>
#include <boost/thread.hpp>
#include <thread>

#include "log_utils.hpp"
#include "motion/leg.h"
#include "utils/dxl_interface.h"
#include "utils/homo.h"
#include "utils/typedefine.h"

/**
 * @brief 重载变胞四足机器人类（爬行）
 * 腿部三个关节轴线方向依次为 Yaw-Roll-Roll
 *
 * @note
 * 1.足式机器人类适用于四足、六足等多足机器人
 * 2.足式机器人类躯干活动度也可以为零；
 * 3.也可以适用于非爬行类机器人类，但是需要改写对应的legs类；
 */
class Quad {
 private:
  QuadParam param_;  // 机器人结构参数
  vector<Leg *> legs_;
  CrimsonParam status_;
  Vec5 waistAngle_;

  Vuc packetID_;
  int startID_;

  vector<Vf> poseList_;  // 逆解计算得到的关节角度值
  vector<vector<vector<vector<int> > > > initPl_;

  YAML::Node cfg_;
  int w_;
  int stepTime_;

  ros::NodeHandle nh_;
  ros::Subscriber enSub_;
  DynamixelInterface dxl_;

  float nds_;  // narrow decrease stride
  float ndw_;  // narrow decrease width
  float ndh_;  // narrow decrease height

  bool quadEnable_;

  void EnCallback(const std_msgs::Bool::ConstPtr &msg);
  /**
   * @brief
   * leg_robot类成员函数，用来根据躯干活动关节角度，求解各条腿全局坐标系到局部坐标系的齐次变换矩阵
   * (计算结果在legs中的tf属性)
   *
   * @param waistAngle 躯干活动关节组成的数组
   */
  void UpdateTF(Vec5 waistAngle);
  /**
   * @brief Set the Local Pos to Legs
   *
   * @param legID 腿号
   * @param localPos 对应的局部齐次坐标
   */
  void SetLocalPos2Leg(vector<int> legID, vector<Vec4> localPos);
  /**
   * @brief Set the Local Pos 2 Leg object
   *
   * @param localPos 全部腿设置为localPos
   */
  void SetLocalPos2Leg(Vec4 localPos);
  /**
   * @brief Set the Global Pos to Legs
   *
   * @param legID 腿号
   * @param globalPos 对应的全局齐次坐标
   */
  void SetGlobalPos2Leg(vector<int> legID, vector<Vec4> globalPos);
  /**
   * @brief
   *
   * @return true
   * @return false
   */
  bool SavePose();
  /**
   * @brief 移动足尖点
   *
   * @param legID 要移动的腿ID
   * @param motion 目标坐标-当前坐标，相对的运动向量
   * @param local 给定的是腿的全局坐标还是局部坐标
   * @param internal false 保存目标位置
   */
  void MoveLeg(int legID, Vec4 motion, bool local, bool internal);
  /**
   * @brief 单腿多次运动
   *
   * @param legID 要移动的腿ID
   * @param motion 相对的运动向量列表
   * @param local 给定的是腿的全局坐标还是局部坐标
   * @param internal false 保存目标位置
   */
  void Leg2MulPos(int legID, vector<Vec4> motion, bool local, bool internal);
  /**
   * @brief 多腿依次做相同的运动
   *
   * @param legID 要移动的腿ID列表
   * @param motion 相对的运动向量
   * @param local 给定的是腿的全局坐标还是局部坐标
   * @param internal false 保存目标位置
   */
  void Legs2Pos(vector<int> legID, Vec4 motion, bool local, bool internal);
  /**
   * @brief 批量移动足尖点
   *
   * @param legID 要移动的腿ID list
   * @param motion 目标坐标-当前坐标，相对的运动向量
   * @param local 给定的是腿的全局坐标还是局部坐标
   * @param internal false 保存目标位置
   */
  void MoveLeg(vector<int> legID, vector<Vec4> motion, bool local,
               bool internal);
  /**
   * @brief 移动身体重心
   *
   * @param motion 目标坐标-当前坐标，相对的运动向量，全局坐标
   * @param internal false 保存目标位置
   */
  void MoveBody(Vec4 motion, bool internal);
  /**
   * @brief
   * 足尖点轨迹为：上升h/2,上升h/2+往外伸展w+往移动方向前进l/2(同时),下降h/2+向内收缩w+往移动方向前进l/2（同时）,下降h/2
   *
   * @param legID 要移动的单个腿ID
   * @param param 三个参数组成的数组，[前进的长度，向外伸展的宽度，迈腿的高度]
   * @param dirfection 前进的方向向量，用来给定迈腿的方向（全局坐标）
   */
  void SwingLeg(int legID, Vec3 param, Vec4 direction);
  /**
   * @brief
   * 足尖点轨迹为：上升h/2,上升h/2+往外伸展w+往移动方向前进l/2(同时),下降h/2+向内收缩w+往移动方向前进l/2（同时）,下降h/2
   *
   * @param legID vec中的腿依次做相同的摆动
   * @param param 三个参数组成的数组，[前进的长度，向外伸展的宽度，迈腿的高度]
   * @param direction 前进的方向向量，用来给定迈腿的方向（全局坐标）
   */
  void SwingLeg(vector<int> legID, Vec3 param, Vec4 direction);
  /**
   * @brief 让某一条腿绕全局坐标系Z轴转动角度, 足尖轨迹与swing一致
   *
   * @param legID 要移动的腿ID
   * @param param 三个参数组成的数组, [1, 向外伸展的宽度，迈腿的高度],
   * 这里的移动的长度通过角度计算得到, 所以这以为直接置1
   * @param theta 转动的角度(全局坐标)
   */
  void RotateLeg(vector<int> legID, Vec3 param, float theta);
  /**
   * @brief
   *
   * @param theta
   */
  void RotateLeg(float theta);
  /**
   * @brief 躯干绕全局坐标系坐标轴转动
   *
   * @param RPY 角度制,
   * 即先绕Y轴转动RPY[0],再绕X轴转动RPY[1],最后绕Z轴转动RPY[2]
   */
  void RotateBody(Vec3 RPY);
  /**
   * @brief
   *
   * @param inverse
   */
  void Run(bool inverse);
  /**
   * @brief
   *
   * @param pose
   */
  void WriteMotor(vector<float> pose);

 public:
  /**
   * @brief Construct a new Quad object
   *
   * @param nh
   */
  explicit Quad(ros::NodeHandle &nh);
  /**
   * @brief Destroy the Quad object
   *
   */
  ~Quad();
  /**
   * @brief 开机或变形后必须调用，更新足式运动状态
   *
   * @param status
   * @param waistAngle
   */
  void UpdateStatus(CrimsonParam status, Vec5 waistAngle);
  /**
   * @brief
   *
   */
  void StandUp();
  /**
   * @brief
   *
   * @param theta 转动角度（角度制）
   * @param h 迈腿高度
   */
  void Turn(float theta, float h);
  /**
   * @brief
   *
   * @param stride 步幅
   * @param h 迈腿高度
   */
  void Go(float stride, float h);
  /**
   * @brief 调整机器人滚转角
   *
   * @param theta 角度制
   */
  void Roll(float theta);
  /**
   * @brief 调整机器人俯仰角
   *
   * @param theta 角度制
   */
  void Pitch(float theta);
};

#endif  // __QUAD_H__