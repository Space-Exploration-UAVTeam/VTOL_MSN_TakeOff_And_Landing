/*
Zhang Bihui @ 20230828
*/

// #pragma once
#include <iostream>
#include <fstream>
#include <cassert>
#include <omp.h>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <thread>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt32.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
// #include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h"
#include "fdilink_ahrs/satellite.h"
#include "fdilink_ahrs/compass.h"
#include "nlink_parser/LinktrackNodeframe2.h"

#include <tf/transform_datatypes.h>
#include <eigen_conversions/eigen_msg.h>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#define D2R (3.14159/180.0)        // degree to radius
#define R2D (180.0/3.14159)        // radius to degree
#define DIM_OF_STATES 9 
#define SKEW_SYM_MATRIX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

ros::Subscriber tag_sub;
ros::Subscriber uwb_sub_0;
ros::Subscriber uwb_sub_1;
ros::Subscriber compass_sub;
ros::Subscriber gnss_gt_sub;

FILE* fp_tag;
std::string filename_tag = "/home/zbh/tag_angle_data.dat"; 
const char* file_name_tag = filename_tag.c_str();
int count = 0;
void tag_callback(const apriltag_ros::AprilTagDetectionArray& tags) //callback still run even if there is no detections!
{
  std::cout<< "tag_callback......" << count++ <<std::endl;
  // double time_tag = tags.header.stamp.toSec();
  int size = tags.detections.size();//多个tag时id顺序不稳定，同时出现哪个更准？取平均！
  if(size == 0)
  {
    return;
  }
  int id;
  Eigen::Affine3d affine;
  Eigen::Matrix4d trans_cam2tag;//tag在相机系下位姿，也是tag系坐标转换到相机系的T
  geometry_msgs::Pose pose_temp;
  pose_temp.position = tags.detections[0].pose.pose.pose.position;
  pose_temp.orientation.x = tags.detections[0].pose.pose.pose.orientation.x;// tag orientation bad! use compass orientation!
  pose_temp.orientation.y = tags.detections[0].pose.pose.pose.orientation.y;
  pose_temp.orientation.z = tags.detections[0].pose.pose.pose.orientation.z;
  pose_temp.orientation.w = tags.detections[0].pose.pose.pose.orientation.w;
  tf::poseMsgToEigen(pose_temp, affine);//
  trans_cam2tag = affine.matrix();
  Eigen::Matrix3d rot_cam2tag;
  rot_cam2tag = trans_cam2tag.block<3,3>(0,0);
  Eigen::Vector3d euler = rot_cam2tag.eulerAngles(2,0,1);//camera downside!!!
  // std::cout<<"yaw,pitch,roll = "<<euler.transpose()<<std::endl;//rpy for UAV
  std::cout<<"yaw = "<< euler(0) <<std::endl;//rpy for UAV

  std::string bufferfile = std::to_string(tags.header.stamp.toSec()) + "," + std::to_string(euler(0)) + "\n";
  const char* buffer_file = bufferfile.c_str();
  std::cout<<"buffer_file: "<< buffer_file <<std::endl;//rpy for UAV
  fwrite(buffer_file, 1, 27, fp_tag);
}

double angle;
FILE* fp_uwb;
std::string filename_uwb = "/home/zbh/uwb_angle_data.dat"; 
const char* file_name_uwb = filename_uwb.c_str();
double x1;
double y_1;
void uwb_callback_1(const nlink_parser::LinktrackNodeframe2& uwb1)
{
  std::cout<<"uwb_callback_1"<<std::endl;
  x1 = uwb1.pos_3d[0];
  y_1 = uwb1.pos_3d[1];
}
void uwb_callback_0(const nlink_parser::LinktrackNodeframe2& uwb0)
{
  std::cout<<"uwb_callback_0"<<std::endl;
  double x0 = uwb0.pos_3d[0];
  double y0 = uwb0.pos_3d[1];
  usleep(100);
  std::cout<<"x1, x0, y1 ,y0 ="<<x1<<","<<x0<<","<<y_1<<","<<y0<<std::endl;
  // std::cout<<fabs(x1-x0)<<std::endl;
  // std::cout<<abs(x1-x0)<<std::endl;
  double angle;
  if(fabs(x1-x0)<0.000001)
  {
    angle = 90;
  }
  else
  {
    angle = atan2((y_1-y0), (x1-x0)) * R2D; 
  }
  std::string bufferfile = std::to_string(uwb0.header.stamp.toSec()) + "," + std::to_string(angle) + "\n";
  const char* buffer_file = bufferfile.c_str();
  std::cout<<"buffer_file: "<< buffer_file <<std::endl;
  fwrite(buffer_file, 1, 30, fp_uwb);
}


FILE* fp_yaw;
std::string filename_yaw = "/home/zbh/yaw_angle_data.dat"; 
const char* file_name_yaw = filename_yaw.c_str();
void compass_callback(const fdilink_ahrs::compass& data_mag)//
{
  std::string bufferfile = std::to_string(data_mag.header.stamp.toSec()) + "," + std::to_string(data_mag.heading) + "\n";
  const char* buffer_file = bufferfile.c_str();
  std::cout<<"buffer_file: "<< buffer_file <<std::endl;
  fwrite(buffer_file, 1, 30, fp_yaw);

}

FILE* fp_alt;
std::string filename_alt = "/home/zbh/alt_data.dat"; 
const char* file_name_alt = filename_alt.c_str();
void gnss_gt_callback(const sensor_msgs::NavSatFix& data)//
{
  std::cout<<"alt_callback"<<std::endl;
  std::string bufferfile = std::to_string(data.header.stamp.toSec()) + "," + std::to_string(data.altitude) + "\n";
  const char* buffer_file = bufferfile.c_str();
  std::cout<<"buffer_file: "<< buffer_file <<std::endl;
  fwrite(buffer_file, 1, 30, fp_alt);
}

int main(int argc, char **argv) 
{
  ros::init(argc, argv, "tag_angle");
  ros::NodeHandle nh("~");
  fp_tag = fopen(file_name_tag,"w");
  fp_uwb = fopen(file_name_uwb,"w");
  fp_yaw = fopen(file_name_yaw,"w");
  fp_alt = fopen(file_name_alt,"w");

  // tag_sub = nh.subscribe("/tag_detections", 10, tag_callback); //【5hz】

  // 出错
  // message_filters::Subscriber<nlink_parser::LinktrackNodeframe2> uwb_sub0(nh, "/nlink_linktrack_nodeframe2_0", 10);//
  // message_filters::Subscriber<nlink_parser::LinktrackNodeframe2> uwb_sub1(nh, "/nlink_linktrack_nodeframe2_1", 10);//
  // typedef message_filters::sync_policies::ApproximateTime<nlink_parser::LinktrackNodeframe2, nlink_parser::LinktrackNodeframe2> MySyncPolicy;
  // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), uwb_sub0, uwb_sub1);
  // message_filters::TimeSynchronizer<nlink_parser::LinktrackNodeframe2, nlink_parser::LinktrackNodeframe2> sync(uwb_sub0, uwb_sub1, 10);
  // sync.registerCallback(boost::bind(&uwb_callback, _1, _2));

  // uwb_sub_0 = nh.subscribe("/nlink_linktrack_nodeframe2_0", 10, uwb_callback_0); //【5hz】      
  // uwb_sub_1 = nh.subscribe("/nlink_linktrack_nodeframe2_1", 10, uwb_callback_1); //【5hz】      

  // compass_sub = nh.subscribe("/mag_pose", 10, compass_callback);//【5hz】 

  gnss_gt_sub = nh.subscribe("/ublox_driver/receiver_lla", 10, gnss_gt_callback);//sensor_msgs/NavSatFixg格式 【10hz】 【差分ublox作为真值】


  ros::spin();
  fclose(fp_tag);
  fclose(fp_uwb);
  fclose(fp_yaw);
  fclose(fp_alt);
}
