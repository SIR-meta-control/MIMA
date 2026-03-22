/*
 * @Author: Yuehao Huang
 * @Date: 2024-04-18 09:49:02
 * @LastEditors: YuehaoHuang yuehaohuang@zju.edu.cn
 * @LastEditTime: 2024-05-19 21:54:55
 * @Description: Publish Color Maps
 */
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

#include "crimson_msgs/Trans.h"

class CloudVisualization {
 private:
  // Subscriber
  ros::Subscriber sub_cloud;
  ros::Subscriber sub_pose;
  ros::Subscriber sub_trans;

  // Publisher
  ros::Publisher pub_color_cloud;
  ros::Publisher pub_color_body_cloud;

  // Published data
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud;

  // Data queue
  std::queue<pcl::PointCloud<pcl::PointXYZRGB>> cloud_queue_;
  std::queue<std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>
      pose_queue_;

  // Synchronized data
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sync_cloud_;
  std::shared_ptr<std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>
      sync_pose_;

  // Extrinsic
  Eigen::Matrix3d R_world_imu;
  Eigen::Vector3d t_world_imu;
  Eigen::Matrix4d T_world_imu;

  Eigen::Matrix3d R_imu_lidar;
  Eigen::Vector3d t_imu_lidar;
  Eigen::Matrix4d T_imu_lidar;

  Eigen::Matrix3d R_lidar_color;
  Eigen::Vector3d t_lidar_color;
  Eigen::Matrix4d T_lidar_color;

  Eigen::Matrix3d R_lidar_color_other;
  Eigen::Vector3d t_lidar_color_other;
  Eigen::Matrix4d T_lidar_color_other;

  Eigen::Matrix3d R_lidar_color_dog;
  Eigen::Vector3d t_lidar_color_dog;
  Eigen::Matrix4d T_lidar_color_dog;

  Eigen::Matrix3d R_color_depth;
  Eigen::Vector3d t_color_depth;
  Eigen::Matrix4d T_color_depth;

  Eigen::Matrix3d R_world_depth;
  Eigen::Vector3d t_world_depth;
  Eigen::Matrix4d T_world_depth;

  // State
  int cfg;

 public:
  CloudVisualization(ros::NodeHandle &nh) {
    sub_cloud = nh.subscribe("/camera/depth/color/points", 1,
                             &CloudVisualization::cloudCallback, this);
    sub_pose = nh.subscribe("/aft_mapped_to_init", 1,
                            &CloudVisualization::poseCallback, this);
    sub_trans = nh.subscribe("/crimson/transform", 1,
                             &CloudVisualization::transCallback, this);

    // Publisher
    pub_color_cloud = nh.advertise<sensor_msgs::PointCloud2>("/color_cloud", 1);
    pub_color_body_cloud =
        nh.advertise<sensor_msgs::PointCloud2>("/color_body_cloud", 1);

    // Initialize
    color_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Extrinsic
    R_imu_lidar << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    t_imu_lidar << 0.04165, 0.02326, -0.0284;
    T_imu_lidar.block<3, 3>(0, 0) = R_imu_lidar;
    T_imu_lidar.block<3, 1>(0, 3) = t_imu_lidar;
    T_imu_lidar.block<1, 4>(3, 0) << 0, 0, 0, 1;

    R_lidar_color_other << -0.01181, 0.00053, 0.99993, -0.99984, 0.01322,
        -0.01182, -0.01322, -0.99991, 0.00038;
    t_lidar_color_other << 0.01077, 0.02414, -0.11213;
    T_lidar_color_other.block<3, 3>(0, 0) = R_lidar_color_other;
    T_lidar_color_other.block<3, 1>(0, 3) = t_lidar_color_other;
    T_lidar_color_other.block<1, 4>(3, 0) << 0, 0, 0, 1;

    R_lidar_color_dog << -0.01345, -0.00176, 0.99991, 0.99991, -0.00176,
        0.01344, 0.00174, 1.0, 0.00178;
    t_lidar_color_dog << 0.05022, -0.09312, -0.01107;
    T_lidar_color_dog.block<3, 3>(0, 0) = R_lidar_color_dog;
    T_lidar_color_dog.block<3, 1>(0, 3) = t_lidar_color_dog;
    T_lidar_color_dog.block<1, 4>(3, 0) << 0, 0, 0, 1;

    R_color_depth << 0.99999, 0.00234, -0.00327, -0.00235, 0.99999, -0.00184,
        0.00326, 0.00185, 0.99999;
    t_color_depth << -0.05919, -0.00013, 0.00049;
    T_color_depth.block<3, 3>(0, 0) = R_color_depth;
    T_color_depth.block<3, 1>(0, 3) = t_color_depth;
    T_color_depth.block<1, 4>(3, 0) << 0, 0, 0, 1;

    cfg = 4;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    // int size = cloud_msg->height * cloud_msg->width;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *current_cloud);
    cloud_queue_.push(*current_cloud);
  }

  void poseCallback(const nav_msgs::OdometryConstPtr &pose_msg) {
    Eigen::Vector3d pos;
    Eigen::Quaterniond q;
    pos = Eigen::Vector3d(pose_msg->pose.pose.position.x,
                          pose_msg->pose.pose.position.y,
                          pose_msg->pose.pose.position.z);
    q = Eigen::Quaterniond(
        pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
        pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z);
    pose_queue_.push(std::make_tuple(pose_msg->header.stamp, pos, q));
  }

  void transCallback(const crimson_msgs::TransConstPtr &trans_msg) {
    cfg = trans_msg->cfg;
  }

  bool Synchronization() {
    if (pose_queue_.empty() || cloud_queue_.empty()) {
      return false;
    }

    ros::Time pose_stamp = std::get<0>(pose_queue_.front());
    ros::Time cloud_stamp =
        ros::Time().fromNSec(cloud_queue_.front().header.stamp * 1000ull);
    ros::Duration duration(0.0001);

    if (pose_stamp < cloud_stamp) {
      while (pose_stamp + duration < cloud_stamp) {
        pose_queue_.pop();
        if (!pose_queue_.empty()) {
          pose_stamp = std::get<0>(pose_queue_.front());
        } else {
          return false;
        }
      }
    } else {
      while (cloud_stamp + duration < pose_stamp) {
        cloud_queue_.pop();
        if (!cloud_queue_.empty()) {
          cloud_stamp =
              ros::Time().fromNSec(cloud_queue_.front().header.stamp * 1000ull);
        } else {
          return false;
        }
      }
    }

    sync_cloud_ = cloud_queue_.front().makeShared();

    sync_pose_ = std::make_shared<
        std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>(
        pose_queue_.front());

    return true;
  }

  void PublishColorCloud() {
    pcl::PointCloud<pcl::PointXYZRGB> color_cloud_frame;
    pcl::PointCloud<pcl::PointXYZRGB> color_body_cloud_frame;

    for (auto &p : *sync_cloud_) {
      if (p.x * p.x + p.y * p.y + p.z * p.z > 4) continue;
      color_body_cloud_frame.push_back(p);
      pointCameraToWorld(p);
      color_cloud_frame.push_back(p);
    }

    *color_cloud += color_cloud_frame;

    sensor_msgs::PointCloud2 color_cloud_msg;
    pcl::toROSMsg(*color_cloud, color_cloud_msg);
    color_cloud_msg.header.frame_id = "camera_init";
    color_cloud_msg.header.stamp =
        ros::Time().fromNSec(sync_cloud_->header.stamp * 1000ull);
    pub_color_cloud.publish(color_cloud_msg);
    color_cloud->clear();

    sensor_msgs::PointCloud2 color_body_cloud_msg;
    pcl::toROSMsg(color_body_cloud_frame, color_body_cloud_msg);
    color_body_cloud_msg.header.frame_id = "camera_init";
    color_body_cloud_msg.header.stamp =
        ros::Time().fromNSec(sync_cloud_->header.stamp * 1000ull);
    pub_color_body_cloud.publish(color_body_cloud_msg);
  }

  void pointCameraToWorld(pcl::PointXYZRGB &p) {
    Eigen::Vector3d p_depth(p.x, p.y, p.z);
    Eigen::Vector3d p_world;

    T_world_imu.block<3, 3>(0, 0) = std::get<2>(*sync_pose_).toRotationMatrix();
    T_world_imu.block<3, 1>(0, 3) = std::get<1>(*sync_pose_);
    T_world_imu.block<1, 4>(3, 0) << 0, 0, 0, 1;

    if (cfg == 3 || cfg == 4) {
      T_lidar_color = T_lidar_color_dog;
    } else {
      T_lidar_color = T_lidar_color_other;
    }

    // T_lidar_color = T_lidar_color_other;

    T_world_depth = T_world_imu * T_imu_lidar * T_lidar_color;
    R_world_depth = T_world_depth.block<3, 3>(0, 0);
    t_world_depth = T_world_depth.block<3, 1>(0, 3);

    p_world = R_world_depth * p_depth + t_world_depth;

    p.x = p_world(0);
    p.y = p_world(1);
    p.z = p_world(2);
  }

  void PopData() {
    cloud_queue_.pop();
    pose_queue_.pop();
  }

  bool Run() {
    if (!Synchronization()) {
      return false;
    }
    PublishColorCloud();
    PopData();
    return true;
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "color_node");
  ros::NodeHandle nh;

  std::shared_ptr<CloudVisualization> cloud_visualization_ =
      std::make_shared<CloudVisualization>(nh);

  ros::Rate rate(100);
  while (ros::ok()) {
    ros::spinOnce();

    cloud_visualization_->Run();

    rate.sleep();
  }

  return 0;
}
