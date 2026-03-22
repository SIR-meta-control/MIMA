#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "initialpose_publisher");
  ros::NodeHandle nh;

  ros::Publisher initialpose_pub =
      nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/initialpose",
                                                             10);

  ros::Duration(2.0).sleep();

  geometry_msgs::PoseWithCovarianceStamped pose_msg;

  pose_msg.header.stamp = ros::Time::now();
  pose_msg.header.frame_id = "camera_init";

  pose_msg.pose.pose.position.x = 0.0;
  pose_msg.pose.pose.position.y = 0.0;
  pose_msg.pose.pose.position.z = 0.0;
  pose_msg.pose.pose.orientation.x = 0.0;
  pose_msg.pose.pose.orientation.y = 0.0;
  pose_msg.pose.pose.orientation.z = 0.0;
  pose_msg.pose.pose.orientation.w = 1.0;

  initialpose_pub.publish(pose_msg);

  ROS_INFO("Initial pose published.");

  ros::spinOnce();

  return 0;
}
