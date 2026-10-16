#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

typedef pcl::PointCloud<pcl::PointXYZI> PointCloudXYZI;

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_publisher");
  ros::NodeHandle nh;

  std::string map_show_path;
  nh.param<std::string>("map_show_path", map_show_path, "");

  PointCloudXYZI::Ptr map_show_cloud(new PointCloudXYZI());
  pcl::io::loadPCDFile<pcl::PointXYZI>(map_show_path, *map_show_cloud);

  ros::Publisher pubLaserCloudMap =
      nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*map_show_cloud, laserCloudmsg);

  laserCloudmsg.header.stamp = ros::Time().now();
  laserCloudmsg.header.frame_id = "camera_init";

  ros::Rate rate(10);
  usleep(1000000);
  for (int i = 0; i < 5; ++i) {
    pubLaserCloudMap.publish(laserCloudmsg);
    rate.sleep();
  }

  return 0;
}
