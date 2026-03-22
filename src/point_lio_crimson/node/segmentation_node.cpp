/*
 * @Author: Yuehao Huang
 * @Date: 2024-04-22 21:43:46
 * @LastEditors: YuehaoHuang yuehaohuang@zju.edu.cn
 * @LastEditTime: 2024-05-16 15:04:57
 * @Description: Publish Segmentation Maps
 */
#include <common_lib.h>
#include <nav_msgs/Odometry.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <boost/filesystem.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

// double segmentation_leaf_size;
// double all_leaf_size;
int pcd_save_interval;

void readParameters(ros::NodeHandle &nh) {
  // nh.param<double>("segmentation_leaf_size", segmentation_leaf_size, 0.01);
  // nh.param<double>("all_leaf_size", all_leaf_size, 0.01);
  nh.param<int>("pcd_save_interval", pcd_save_interval, 3);
}

void DownsamplePointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud,
                          double leaf_size) {
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_grid_filter.setInputCloud(input_cloud);
  voxel_grid_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_grid_filter.filter(*output_cloud);
}

class CloudSegmentation {
 private:
  // Subscriber
  ros::Subscriber sub_cloud;
  ros::Subscriber sub_pose;

  // Publisher
  // ros::Publisher pub_segmentation_cloud;
  ros::Publisher pub_map_cloud;

  // Published data
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud;
  // pcl::PointCloud<pcl::PointXYZI>::Ptr segmentation_cloud;

  // Data queue
  std::queue<pcl::PointCloud<pcl::PointXYZI>> cloud_queue_;
  std::queue<std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>
      pose_queue_;

  // Synchronized data
  pcl::PointCloud<pcl::PointXYZI>::Ptr sync_cloud_;
  std::shared_ptr<std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>
      sync_pose_;
  std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond> last_sync_pose_;

  // threshold
  double angle_threshold_;
  ros::Duration time_threshold_;
  double distance_threshold_;

  // pcl
  int pcd_index;

  // stringstream
  // std::stringstream map_cloud_ss;
  // std::stringstream segmentation_cloud_ss;

 public:
  CloudSegmentation(ros::NodeHandle &nh) {
    // Subscriber
    // sub_cloud = nh.subscribe("/cloud_registered_body", 1,
    //                          &CloudSegmentation::cloudCallback, this);
    sub_cloud = nh.subscribe("/cloud_registered", 1,
                             &CloudSegmentation::cloudCallback, this);
    sub_pose = nh.subscribe("/aft_mapped_to_init", 1,
                            &CloudSegmentation::poseCallback, this);

    // Publisher
    // pub_segmentation_cloud =
    //     nh.advertise<sensor_msgs::PointCloud2>("/segmentation_cloud", 1);
    pub_map_cloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 1);

    // Initialize
    // segmentation_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    map_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    sync_cloud_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    sync_pose_ = std::make_shared<
        std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>();

    // threshold
    angle_threshold_ = 0.5;
    time_threshold_ = ros::Duration(0.5);
    distance_threshold_ = 0.5;

    pcd_index = 0;

    // map_cloud_ss << std::fixed << std::setprecision(2) << all_leaf_size;
    // segmentation_cloud_ss << std::fixed << std::setprecision(2)
    //                       << segmentation_leaf_size;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    int size = cloud_msg->height * cloud_msg->width;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
        new pcl::PointCloud<pcl::PointXYZI>(size, 1));
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

  bool Synchronization() {
    if (pose_queue_.empty() || cloud_queue_.empty()) {
      return false;
    }

    ros::Time pose_stamp = std::get<0>(pose_queue_.front());
    ros::Time cloud_stamp =
        ros::Time().fromNSec(cloud_queue_.front().header.stamp * 1000ull);
    ros::Duration duration(0.000001);

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

    // sync_cloud_ = cloud_queue_.front();
    sync_cloud_ = cloud_queue_.front().makeShared();
    // sync_cloud_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(
    //     cloud_queue_.front());

    sync_pose_ = std::make_shared<
        std::tuple<ros::Time, Eigen::Vector3d, Eigen::Quaterniond>>(
        pose_queue_.front());

    return true;
  }

  void PublishSegmentationCloud() {
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_frame(
        new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr segmentation_cloud_frame(
    //     new pcl::PointCloud<pcl::PointXYZI>);

    ros::Duration time_diff =
        std::get<0>(*sync_pose_) - std::get<0>(last_sync_pose_);
    double distance_diff =
        (std::get<1>(*sync_pose_) - std::get<1>(last_sync_pose_)).norm();
    double angle_diff =
        2.0 *
        std::acos(std::abs(
            (std::get<2>(*sync_pose_) * std::get<2>(last_sync_pose_).inverse())
                .w())) *
        180.0 / M_PI;

    // Determine if it is a keyframe
    if (true || (time_diff > time_threshold_) ||
        (distance_diff > distance_threshold_) ||
        (angle_diff > angle_threshold_)) {
      // pointIMUToWorld(all_cloud_frame);
      for (int i = 0; i < sync_cloud_->size(); i++) {
        map_cloud_frame->push_back(sync_cloud_->points[i]);
      }
      // FilterPointCloud(map_cloud_frame, segmentation_cloud_frame);

      // DownsamplePointCloud(all_cloud_frame, all_cloud_frame, all_leaf_size);

      // DownsamplePointCloud(segmentation_cloud_frame,
      // segmentation_cloud_frame,
      //                      segmentation_leaf_size);

      *map_cloud += *map_cloud_frame;
      // *segmentation_cloud += *segmentation_cloud_frame;

      last_sync_pose_ = *sync_pose_;

      // publish segmentation cloud
      // sensor_msgs::PointCloud2 segmentation_cloud_msg;
      // pcl::toROSMsg(*segmentation_cloud, segmentation_cloud_msg);
      // segmentation_cloud_msg.header.frame_id = "camera_init";
      // segmentation_cloud_msg.header.stamp =
      //     ros::Time().fromNSec(sync_cloud_->header.stamp * 1000ull);
      // pub_segmentation_cloud.publish(segmentation_cloud_msg);

      sensor_msgs::PointCloud2 map_cloud_msg;
      pcl::toROSMsg(*map_cloud, map_cloud_msg);
      map_cloud_msg.header.frame_id = "camera_init";
      map_cloud_msg.header.stamp =
          ros::Time().fromNSec(sync_cloud_->header.stamp * 1000ull);
      pub_map_cloud.publish(map_cloud_msg);

      // save pcd frame
      SavePCD();
    }
  }

  void SavePCD() {
    static int wait_num = 0;
    wait_num++;
    if (map_cloud->size() > 0 && pcd_save_interval > 0 &&
        wait_num >= pcd_save_interval) {
      pcd_index++;

      string pcd_dir(string(ROOT_DIR) + "PCD/");
      boost::filesystem::path directory_path(pcd_dir);
      if (!boost::filesystem::exists(directory_path)) {
        boost::filesystem::create_directories(directory_path);
      }

      // string segmentation_cloud_dir(pcd_dir + "/segmentation_" +
      //                               to_string(pcd_index) + ".pcd");
      string map_cloud_dir(pcd_dir + "/map_" + to_string(pcd_index) + ".pcd");

      pcl::PCDWriter pcd_writer;

      // pcd_writer.writeBinary(segmentation_cloud_dir, *segmentation_cloud);
      pcd_writer.writeBinary(map_cloud_dir, *map_cloud);

      std::cout << "Save pcd: " << map_cloud_dir << std::endl;

      // segmentation_cloud->clear();
      map_cloud->clear();

      wait_num = 0;
    }
  }

  void PopData() {
    cloud_queue_.pop();
    pose_queue_.pop();
  }

  bool Run() {
    if (!Synchronization()) {
      return false;
    }
    PublishSegmentationCloud();
    PopData();
    return true;
  }

  // void pointIMUToWorld(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_frame) {
  //   Eigen::Matrix3d R = std::get<2>(*sync_pose_).toRotationMatrix();
  //   Eigen::Vector3d t = std::get<1>(*sync_pose_);
  //   for (int i = 0; i < sync_cloud_->size(); i++) {
  //     Eigen::Vector3d p(sync_cloud_->points[i].x, sync_cloud_->points[i].y,
  //                       sync_cloud_->points[i].z);
  //     p = R * p + t;
  //     sync_cloud_->points[i].x = p.x();
  //     sync_cloud_->points[i].y = p.y();
  //     sync_cloud_->points[i].z = p.z();

  //     cloud_frame->push_back(sync_cloud_->points[i]);
  //   }
  // }

  void FilterPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud) {
    for (const auto &point : *input_cloud) {
      if (point.z < (std::get<1>(*sync_pose_).z() + 0.2) &&
          point.intensity > 0.1) {
        output_cloud->push_back(point);
      }
    }
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "segmentation_node");
  ros::NodeHandle nh;

  readParameters(nh);
  std::shared_ptr<CloudSegmentation> cloud_segmentation_ =
      std::make_shared<CloudSegmentation>(nh);

  ros::Rate rate(100);
  while (ros::ok()) {
    ros::spinOnce();

    cloud_segmentation_->Run();

    rate.sleep();
  }

  return 0;
}
