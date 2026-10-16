/*
 * @Author: Yuehao Huang
 * @Date: 2024-04-25 15:58:39
 * @LastEditors: YuehaoHuang yuehaohuang@zju.edu.cn
 * @LastEditTime: 2024-05-16 11:26:01
 * @Description: combine point clouds
 */

#include <common_lib.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

double map_leaf_size;
double mean_k;
double stddev_mul_thresh;
double min_x;
double max_x;
double min_y;
double max_y;
double min_z;
double max_z;
bool roi_en;

void ReadParameters(ros::NodeHandle &nh) {
  nh.param<double>("map_leaf_size", map_leaf_size, 0.3);
  nh.param<double>("mean_k", mean_k, 20);
  nh.param<double>("stddev_mul_thresh", stddev_mul_thresh, 1.0);
  nh.param<bool>("roi_en", roi_en, false);
  nh.param<double>("min_x", min_x, -15.0);
  nh.param<double>("max_x", max_x, 13.0);
  nh.param<double>("min_y", min_y, -20.0);
  nh.param<double>("max_y", max_y, 25.0);
  nh.param<double>("min_z", min_z, -2.0);
  nh.param<double>("max_z", max_z, 3.0);
}

void GetROICloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud,
                 double min_x, double max_x, double min_y, double max_y,
                 double min_z, double max_z) {
  pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(
      new pcl::ConditionAnd<pcl::PointXYZI>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT,
                                               min_x)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT,
                                               max_x)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("y", pcl::ComparisonOps::GT,
                                               min_y)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("y", pcl::ComparisonOps::LT,
                                               max_y)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::GT,
                                               min_z)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::LT,
                                               max_z)));

  pcl::ConditionalRemoval<pcl::PointXYZI> condrem;
  condrem.setCondition(range_cond);
  condrem.setInputCloud(input_cloud);
  condrem.setKeepOrganized(true);
  condrem.filter(*output_cloud);
}

void RemoveOutliersCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud,
                         int mean_k, double stddev_mul_thresh) {
  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
  sor.setInputCloud(input_cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(stddev_mul_thresh);

  sor.filter(*output_cloud);
}

template <typename PointT>
void DownsampleCloudAdapted(pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out,
                            double leaf_size) {
  cloud_out.clear();

  Eigen::Vector4f min_p;  // 用于存放三个轴的最小值
  Eigen::Vector4f max_p;  // 用于存放三个轴的最大值
  pcl::getMinMax3D(cloud_in, min_p, max_p);

  std::int64_t dx, dy, dz;

  double temp_dx = (max_p[0] - min_p[0]) / leaf_size;
  double temp_dy = (max_p[1] - min_p[1]) / leaf_size;
  double temp_dz = (max_p[2] - min_p[2]) / leaf_size;

  /// 自动切分
  Eigen::Vector3d splite_num(1, 1, 1);
  while (true) {
    dx = static_cast<std::int64_t>(temp_dx / splite_num(0)) + 1;
    dy = static_cast<std::int64_t>(temp_dy / splite_num(1)) + 1;
    dz = static_cast<std::int64_t>(temp_dz / splite_num(2)) + 1;

    if ((dx * dy * dz) <
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
      break;
    }

    if (dx > dy) {
      splite_num(0) += 1;
    } else {
      splite_num(1) += 1;
    }
  }

  //  std::cout << "Splite num : " << splite_num.transpose() << std::endl;

  double stepX = (max_p[0] - min_p[0]) / splite_num(0);
  double stepY = (max_p[1] - min_p[1]) / splite_num(1);

  for (int i = 0; i < splite_num(0); i++) {
    for (int j = 0; j < splite_num(1); j++) {
      // 获得子区域中的点云
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud_in.makeShared());
      pass.setFilterFieldName("x");
      pass.setFilterLimits(
          min_p[0] + i * stepX,
          min_p[0] + i * stepX + stepX);  // 保留或过滤z轴方向-1.2到0
      // pass.setFilterLimitsNegative(true);//设置过滤器限制负//设置保留范围内false
      pcl::PointCloud<PointT> cloud_filtered;
      pass.filter(cloud_filtered);

      pass.setInputCloud(cloud_filtered.makeShared());
      pass.setFilterFieldName("y");
      pass.setFilterLimits(
          min_p[1] + j * stepY,
          min_p[1] + j * stepY + stepY);  // 保留或过滤z轴方向-1.2到0
                                          //  VPointCloud cloud_filtered;
      pass.filter(cloud_filtered);

      pcl::VoxelGrid<PointT> sor;
      sor.setInputCloud(cloud_filtered.makeShared());
      sor.setLeafSize((float)leaf_size, (float)leaf_size, (float)leaf_size);
      pcl::PointCloud<PointT> cloud_downsample;
      sor.filter(cloud_downsample);

      cloud_out += cloud_downsample;
    }
  }
}

void LoadAndCombinePointCloud(
    const std::string &directory,
    pcl::PointCloud<pcl::PointXYZI>::Ptr combined_map_cloud) {
  boost::filesystem::path pcd_dir(directory);

  for (const auto &entry : boost::filesystem::directory_iterator(pcd_dir)) {
    if (boost::filesystem::is_regular_file(entry.status())) {
      std::string file_name = entry.path().filename().string();

      if (file_name.find("map_") == 0 &&
          file_name.substr(file_name.find_last_of(".") + 1) == "pcd") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(
            new pcl::PointCloud<pcl::PointXYZI>);
        std::cout << entry.path().string() << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(entry.path().string(),
                                                 *map_cloud) == -1) {
          std::cout << "Error loading file: " << file_name << std::endl;
          continue;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_xyz(
            new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*map_cloud, *map_cloud_xyz);

        *combined_map_cloud += *map_cloud_xyz;
      }
    }
  }

}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "combine_node");
  ros::NodeHandle nh;

  ReadParameters(nh);

  pcl::PCDWriter pcd_writer;

  std::stringstream map_cloud_ss;

  map_cloud_ss << std::fixed << std::setprecision(2) << map_leaf_size;

  pcl::PointCloud<pcl::PointXYZI>::Ptr combined_map_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr combined_map_downsampled_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr combined_map_removeoutliers_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr combined_map_roi_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  string pcd_dir(string(ROOT_DIR) + "PCD/");
  std::cout << "PCD dir: " << pcd_dir << std::endl;
  boost::filesystem::path directory_path(pcd_dir + "/results");
  if (!boost::filesystem::exists(directory_path)) {
    boost::filesystem::create_directories(directory_path);
  }

  std::cout << "Combining point clouds from: " << pcd_dir << std::endl;

  string combined_map_dir(pcd_dir + "/results/map.pcd");
  string combined_map_downsampled_dir(pcd_dir + "/results/map_downsampled.pcd");
  string combined_map_removeoutliers_dir(pcd_dir +
                                         "/results/map_removeoutliers.pcd");
  string combined_map_roi_dir(pcd_dir + "/results/map_roi.pcd");

  LoadAndCombinePointCloud(pcd_dir, combined_map_cloud);

  DownsampleCloudAdapted(*combined_map_cloud,
  *combined_map_downsampled_cloud,
                         map_leaf_size);
  RemoveOutliersCloud(combined_map_downsampled_cloud,
                      combined_map_removeoutliers_cloud, mean_k,
                      stddev_mul_thresh);

  pcd_writer.writeBinary(combined_map_dir, *combined_map_cloud);
  pcd_writer.writeBinary(combined_map_downsampled_dir,
                         *combined_map_downsampled_cloud);
  pcd_writer.writeBinary(combined_map_removeoutliers_dir,
                         *combined_map_removeoutliers_cloud);

  if (roi_en) {
    GetROICloud(combined_map_removeoutliers_cloud, combined_map_roi_cloud,
                min_x, max_x, min_y, max_y, min_z, max_z);
    pcd_writer.writeBinary(combined_map_roi_dir, *combined_map_roi_cloud);
  }

  return 0;
}
