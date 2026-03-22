#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>
#include <sstream>

cv::Mat latest_image;
ros::Time latest_stamp;
bool new_image_available = false;

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  try {
    // 将ROS图像消息转换为OpenCV图像
    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    latest_image = cv_ptr->image;
    latest_stamp = msg->header.stamp;
    new_image_available = true;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}

void timerCallback(const ros::TimerEvent&) {
  if (new_image_available) {
    // 生成文件名并保存图像
    std::stringstream ss;
    ss << latest_stamp.sec << "_" << latest_stamp.nsec;
    std::string filedir = std::string(ROOT_DIR);
    std::string filename = filedir + "jpg/outdoor/" + ss.str() + ".png";
    cv::imwrite(filename, latest_image);

    ROS_INFO("Saved image to %s", filename.c_str());

    new_image_available = false;  // 重置标志
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "image_saver");
  ros::NodeHandle nh;

  // 订阅图像话题
  ros::Subscriber sub =
      nh.subscribe("/camera/depth/image_rect_raw", 10, imageCallback);

  // 创建一个定时器，每0.5秒保存一次图像
  ros::Timer timer = nh.createTimer(ros::Duration(0.5), timerCallback);

  ros::spin();
  return 0;
}
