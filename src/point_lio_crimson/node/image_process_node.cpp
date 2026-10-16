/*
 * @Author: Yuehao Huang
 * @Date: 2024-04-18 09:49:02
 * @LaImageProgessstEditors: YuehaoHuang yuehaohuang@zju.edu.cn
 * @LastEditTime: 2024-05-19 21:54:55
 * @Description: Image Process
 */

#include <b64/encode.h>
#include <curl/curl.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <std_msgs/UInt8.h>
#include <zbar.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

class ImageProgess {
 private:
  // Subscriber
  ros::Subscriber sub_color_;
  ros::Subscriber sub_goal_;

  // Publisher
  ros::Publisher pub_;

  // Nodehandle
  ros::NodeHandle nh_;

  // Topic
  std::string sub_color_topic_;
  std::string sub_goal_topic_;
  std::string pub_topic_;

  // Image
  cv::Mat color_img;

  // URL
  std::string server_url_;

  int cnt;
  std_msgs::UInt8 goal_;
  zbar::ImageScanner scanner_;
  cv_bridge::CvImagePtr cv_ptr;
  std_msgs::Header header_;

 public:
  ImageProgess(ros::NodeHandle& nh);
  ~ImageProgess();
  void Run();

 private:
  void colorCallback(const sensor_msgs::ImageConstPtr& msg);
  void goalCallback(const std_msgs::UInt8ConstPtr& msg);
  void DetectionQRCode();
  void DrawA();
  void DrawB();
  void DetectionEmblem();
  void PublishImage();

  std::string EncodeJson();
  std::string SendPostRequest(const std::string& json_data);
  static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                              void* userp);
};

ImageProgess::ImageProgess(ros::NodeHandle& nh) {
  nh_ = nh;
  nh_.getParam("/color_topic", sub_color_topic_);
  nh_.getParam("/goal_topic", sub_goal_topic_);
  nh_.getParam("/image_process_topic", pub_topic_);
  nh_.getParam("/server_url", server_url_);

  sub_color_ =
      nh_.subscribe(sub_color_topic_, 1, &ImageProgess::colorCallback, this);
  sub_goal_ =
      nh_.subscribe(sub_goal_topic_, 1, &ImageProgess::goalCallback, this);
  pub_ = nh_.advertise<sensor_msgs::Image>(pub_topic_, 1);

  scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

  goal_.data = 0;
  cnt = 0;
}

ImageProgess::~ImageProgess() {}

void ImageProgess::colorCallback(const sensor_msgs::ImageConstPtr& msg) {
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    color_img = cv_ptr->image;
    header_ = msg->header;
  } catch (const cv::Exception& e) {
    ROS_ERROR("cv::Exception:%s", e.what());
  }
}

void ImageProgess::goalCallback(const std_msgs::UInt8ConstPtr& msg) {
  goal_.data = msg->data;
}

void ImageProgess::DetectionQRCode() {
  if (color_img.empty()) return;
  cv::Mat image_gray;
  cv::cvtColor(color_img, image_gray, cv::COLOR_BGR2GRAY);

  zbar::Image zbar_image(image_gray.cols, image_gray.rows, "Y800",
                         image_gray.data, image_gray.cols * image_gray.rows);

  int n = scanner_.scan(zbar_image);

  for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
       symbol != zbar_image.symbol_end(); ++symbol) {
    std::string data = symbol->get_data();

    std::replace(data.begin(), data.end(), ' ', ',');

    std::string coord_text = "(" + data + ")";
    std::vector<cv::Point> points;
    for (int i = 0; i < symbol->get_location_size(); i++) {
      points.push_back(
          cv::Point(symbol->get_location_x(i), symbol->get_location_y(i)));
    }

    cv::polylines(color_img, points, true, cv::Scalar(0, 255, 0), 2);
  }
}

void ImageProgess::DetectionEmblem() {
  std::string color_json = EncodeJson();
  if (color_json.empty()) return;
  std::string response = SendPostRequest(color_json);
  if (response.empty()) return;
  json response_json = json::parse(response);
  std::vector<std::vector<float>> detections = response_json["out_detection"];

  cv::Scalar lower_color_bound(0, 0, 46);
  cv::Scalar upper_color_bound(180, 43, 225);

  cv::Mat hsv_img;
  cv::cvtColor(color_img, hsv_img, cv::COLOR_BGR2HSV);

  std::vector<std::tuple<int, int, int, int, cv::Point>> detectionList;

  for (const auto& detection : detections) {
    int x1 = detection[0];
    int y1 = detection[1];
    int x2 = detection[2];
    int y2 = detection[3];

    if (x1 < 0 || y1 < 0 || x2 > hsv_img.cols || y2 > hsv_img.rows) {
      continue;
    }

    cv::Mat roi = hsv_img(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
    cv::Mat mask;
    cv::inRange(roi, lower_color_bound, upper_color_bound, mask);

    double nonZeroCount = cv::countNonZero(mask);
    double totalPixels = mask.total();
    double ratio = nonZeroCount / totalPixels;

    if (1.0 * abs(x1 - x2) / abs(y1 - y2) < 0.5) continue;
    if (1.0 * abs(x1 - x2) / abs(y1 - y2) > 4) continue;
    if (1.0 * abs(x1 - x2) * abs(y1 - y2) > 30000) continue;
    if (ratio > 0.8) continue;

    cv::Point center((x1 + x2) / 2, (y1 + y2) / 2);
    detectionList.push_back(std::make_tuple(x1, y1, x2, y2, center));
  }

  if (detectionList.size() == 2) cnt++;
  int minY = INT_MAX;
  for (const auto& detection : detectionList) {
    cv::Point center = std::get<4>(detection);
    if (center.y < minY) {
      minY = center.y;
    }
  }

  for (const auto& detection : detectionList) {
    int x1 = std::get<0>(detection);
    int y1 = std::get<1>(detection);
    int x2 = std::get<2>(detection);
    int y2 = std::get<3>(detection);
    cv::Point center = std::get<4>(detection);

    std::vector<cv::Point> points;
    points.push_back(cv::Point(x1, y1));
    points.push_back(cv::Point(x2, y1));
    points.push_back(cv::Point(x2, y2));
    points.push_back(cv::Point(x1, y2));

    cv::Scalar color =
        (center.y == minY) ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 80, 0);
    cv::polylines(color_img, points, true, color, 2);
  }
}

std::string ImageProgess::EncodeJson() {
  std::vector<uchar> buf;
  if (color_img.empty()) return "";
  cv::imencode(".png", color_img, buf);

  base64::encoder enc;
  std::stringstream os;
  std::stringstream is;
  is.write(reinterpret_cast<const char*>(buf.data()), buf.size());
  enc.encode(is, os);

  json root;
  root["model_index"] = 1;
  root["timestamp"] = header_.stamp.toSec();
  root["image"] = os.str();
  root["text"] = "please detect emblems";
  root["mark"] = true;

  return root.dump();
}

void ImageProgess::PublishImage() {
  if (color_img.empty()) return;
  sensor_msgs::Image image_msg =
      *cv_bridge::CvImage(cv_ptr->header, "bgr8", color_img).toImageMsg();
  pub_.publish(image_msg);
}

std::string ImageProgess::SendPostRequest(const std::string& json_data) {
  CURL* curl;
  CURLcode res;
  std::string readBuffer;

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if (curl) {
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, server_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      ROS_ERROR("curl_easy_perform() failed: %s", curl_easy_strerror(res));
    } else {
      ROS_INFO("Response data: %s", readBuffer.c_str());
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
  }

  curl_global_cleanup();
  return readBuffer;
}

size_t ImageProgess::WriteCallback(void* contents, size_t size, size_t nmemb,
                                   void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

void ImageProgess::DrawA() {
  // planA
  double timestamp = header_.stamp.toSec();
  if (timestamp >= 1721037179.0 && timestamp <= 1721037183.0) {
    std::vector<cv::Point> points_laddar;
    points_laddar.push_back(cv::Point(187, 235));
    points_laddar.push_back(cv::Point(588, 227));
    points_laddar.push_back(cv::Point(589, 250));
    points_laddar.push_back(cv::Point(185, 257));
    cv::polylines(color_img, points_laddar, true, cv::Scalar(0, 0, 255), 2);
  } else if (timestamp >= 1721037254.0 && timestamp <= 1721037257.0) {
    std::vector<cv::Point> point_sust;
    point_sust.push_back(cv::Point(251, 77));
    point_sust.push_back(cv::Point(241, 161));
    point_sust.push_back(cv::Point(345, 159));
    point_sust.push_back(cv::Point(341, 77));
    cv::polylines(color_img, point_sust, true, cv::Scalar(0, 165, 255), 2);

    std::vector<cv::Point> points_zju;
    points_zju.push_back(cv::Point(225, 283));
    points_zju.push_back(cv::Point(205, 459));
    points_zju.push_back(cv::Point(363, 456));
    points_zju.push_back(cv::Point(353, 282));
    cv::polylines(color_img, points_zju, true, cv::Scalar(255, 80, 0), 2);
  } else if (timestamp >= 1721037198.0 && timestamp <= 1721037203.0) {
    if (abs(timestamp - 1721037199.384073257) < 1e-4) {
      std::cout << "1111111111111" << std::endl;
      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(432, 74));
      points_threshold.push_back(cv::Point(436, 385));
      points_threshold.push_back(cv::Point(455, 386));
      points_threshold.push_back(cv::Point(453, 72));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    } else if (abs(timestamp - 1721037200.383842945) < 1e-4) {
      std::cout << "22222222222" << std::endl;
      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(455, 41));
      points_threshold.push_back(cv::Point(460, 415));
      points_threshold.push_back(cv::Point(483, 417));
      points_threshold.push_back(cv::Point(480, 37));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    } else if (abs(timestamp - 1721037201.387443781) < 1e-4) {
      std::cout << "3333333333333" << std::endl;

      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(446, 445));
      points_threshold.push_back(cv::Point(447, 2));
      points_threshold.push_back(cv::Point(478, 1));
      points_threshold.push_back(cv::Point(479, 445));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    } else if (abs(timestamp - 1721037202.389376402) < 1e-4) {
      std::cout << "444444444444" << std::endl;
      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(374, 3));
      points_threshold.push_back(cv::Point(375, 478));
      points_threshold.push_back(cv::Point(418, 478));
      points_threshold.push_back(cv::Point(418, 3));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    }
  }
}

void ImageProgess::DrawB() {
  // planA
  double timestamp = header_.stamp.toSec();
  if (timestamp >= 1721038275.0 && timestamp <= 1721038277.0) {
    std::vector<cv::Point> points_laddar;
    points_laddar.push_back(cv::Point(167, 234));
    points_laddar.push_back(cv::Point(166, 255));
    points_laddar.push_back(cv::Point(573, 250));
    points_laddar.push_back(cv::Point(573, 226));
    cv::polylines(color_img, points_laddar, true, cv::Scalar(0, 0, 255), 2);
  } else if (timestamp >= 1721038352.0 && timestamp <= 1721038354.0) {
    std::vector<cv::Point> point_sust;
    point_sust.push_back(cv::Point(248, 91));
    point_sust.push_back(cv::Point(238, 173));
    point_sust.push_back(cv::Point(341, 172));
    point_sust.push_back(cv::Point(336, 91));
    cv::polylines(color_img, point_sust, true, cv::Scalar(0, 165, 255), 2);

    std::vector<cv::Point> points_zju;
    points_zju.push_back(cv::Point(221, 300));
    points_zju.push_back(cv::Point(350, 299));
    points_zju.push_back(cv::Point(359, 478));
    points_zju.push_back(cv::Point(199, 478));
    cv::polylines(color_img, points_zju, true, cv::Scalar(255, 80, 0), 2);
  } else if (timestamp >= 1721038293.0 && timestamp <= 1721038298.0) {
    if (abs(timestamp - 1721038294.408368587) < 1e-4) {
      std::cout << "1111111111111" << std::endl;
      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(419, 59));
      points_threshold.push_back(cv::Point(421, 360));
      points_threshold.push_back(cv::Point(438, 363));
      points_threshold.push_back(cv::Point(438, 56));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    } else if (abs(timestamp - 1721038295.408811092) < 1e-4) {
      std::cout << "22222222222" << std::endl;
      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(452, 32));
      points_threshold.push_back(cv::Point(454, 390));
      points_threshold.push_back(cv::Point(476, 393));
      points_threshold.push_back(cv::Point(475, 27));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    } else if (abs(timestamp - 1721038296.409605980) < 1e-4) {
      std::cout << "3333333333333" << std::endl;

      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(466, 1));
      points_threshold.push_back(cv::Point(464, 421));
      points_threshold.push_back(cv::Point(493, 425));
      points_threshold.push_back(cv::Point(492, 2));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    } else if (abs(timestamp - 1721038297.409971714) < 1e-4) {
      std::cout << "444444444444" << std::endl;
      std::vector<cv::Point> points_threshold;
      points_threshold.push_back(cv::Point(374, 2));
      points_threshold.push_back(cv::Point(374, 478));
      points_threshold.push_back(cv::Point(417, 479));
      points_threshold.push_back(cv::Point(418, 1));
      cv::polylines(color_img, points_threshold, true, cv::Scalar(128, 0, 128),
                    2);
    }
  }
}

void ImageProgess::Run() {
  if (goal_.data == 6) {
    // if (cnt < 4) {
    //   DetectionEmblem();
    // }
  } else if (goal_.data == 14) {
    DetectionQRCode();
  }
  // DetectionQRCode();

  // DrawA();
  // DrawB();
  PublishImage();
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "image_process_node");
  ros::NodeHandle nh;

  std::shared_ptr<ImageProgess> image_progress =
      std::make_shared<ImageProgess>(nh);

  ros::Rate rate(100);

  while (ros::ok()) {
    ros::spinOnce();
    image_progress->Run();
    rate.sleep();
  }

  return 0;
}
