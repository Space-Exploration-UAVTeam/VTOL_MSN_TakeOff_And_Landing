/*
Zhang Bihui @ 20230605
for VTOL aircraft Multi-Sensor Navigation in taking off or landing：gnss + compass + imu + apriltag + lidar pointcould registration + uwb
with: Error State Kalman Filter
StatesGroup:
  [0-2] the estimated attitude
  [3-5] the estimated position
  [6-8] the estimated velocity
传感器噪声项：固定值
*/

// #pragma once
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <omp.h>
#include <mutex>
// #include <math.h>
#include <cmath>
#include <thread>
// #include <shared_mutex>
// #include <csignal>
// #include <unistd.h>
// #include <so3_math.h>//RotMtoEuler
// #include <common_lib.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
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

#define OUTPUT_FOR_PAPER
// #define LOW_PASS_FILTER
#define D2R (3.14159/180.0)        // degree to radius
#define R2D (180.0/3.14159)        // radius to degree
#define DIM_OF_STATES 9 
#define SKEW_SYM_MATRIX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

/*------------------------------------------------控制参数---------------------------------------------*/
int threshold_anchor_pos_ = 2;//参与计算 anchor_transition 的数据量阈值
int threshold_anchor_yaw_ = 2;//参与计算 anchor_transition 的数据量阈值
int threshold_gnss_star_ = 10;//gnss搜星数量阈值
//噪声
double R_init_COV = 0.0001;//初始P，P大有利于快速收敛？
double P_init_COV = 0.0001;
double V_init_COV = 0.0001;

double COV_OMEGA_NOISE_DIAG = 0.001;//【角速度误差=陀螺仪误差】
double COV_VEL_NOISE_DIAG = 0.002; //【速度误差】
double COV_ACC_NOISE_DIAG = 0.004;  //【加速度误差】
double GNSS_OBSERVE_COV = 0.0005;  //观测方程噪声，收敛后传感器的逆权重
double COMPASS_OBSERVE_COV = 0.0005;
double TAG_OBSERVE_COV = 0.0005;
double UWB_OBSERVE_COV = 0.0005;
//NDT参数
static float map_voxel_size_ = 0.5; ////////////////////////////////<m>
static float scan_voxel_size_ = 0.2;////////////////////////////////<m>
static float distance_threshold_ = 10.0;//<m?>
static float probability_threshold_ = 0.75;//////////////////////////////<什么单位？？？>
static float lidar_range_ = 50;//<m>
static float ndt_epsilon_ = 0.01;
static float ndt_step_size_ = 0.1;
static float ndt_resolution_ = 1.0;
static float ndt_iterations_ = 100;
static float ndt_num_threads_ = 6; 
#ifdef LOW_PASS_FILTER
//滤波器截止频率
double cutoff_freq_ = 20;
/*------------------------------------------------2nd order butterworth LowPassFilter---------------------------------------------*/
class LowPassFilter2pVector3d
{
public:
	LowPassFilter2pVector3d(double sample_freq, double cutoff_freq)
	{
		set_cutoff_frequency(sample_freq, cutoff_freq);
	}

	void set_cutoff_frequency(double sample_freq, double cutoff_freq)
  {
    _cutoff_freq = cutoff_freq;
    // reset delay elements on filter change
    _delay_element_1.setZero();
    _delay_element_2.setZero();
    if (_cutoff_freq <= 0.0f) {
      // no filtering
      _b0 = 1.0f;
      _b1 = 0.0f;
      _b2 = 0.0f;
      _a1 = 0.0f;
      _a2 = 0.0f;
      return;
    }
    const double fr = sample_freq / _cutoff_freq;
    const double ohm = tanf(3.14159 / fr);
    const double c = 1.0f + 2.0f * cosf(3.14159 / 4.0f) * ohm + ohm * ohm;
    _b0 = ohm * ohm / c;
    _b1 = 2.0f * _b0;
    _b2 = _b0;
    _a1 = 2.0f * (ohm * ohm - 1.0f) / c;
    _a2 = (1.0f - 2.0f * cosf(3.14159 / 4.0f) * ohm + ohm * ohm) / c;
	}

	inline Eigen::Vector3d apply(const Eigen::Vector3d &sample)
	{
		// do the filtering
		const Eigen::Vector3d delay_element_0(sample - _delay_element_1 *_a1 - _delay_element_2 * _a2);
		const Eigen::Vector3d output(delay_element_0 *_b0 + _delay_element_1 *_b1 + _delay_element_2 * _b2);
		_delay_element_2 = _delay_element_1;
		_delay_element_1 = delay_element_0;
		return output;
    }

private:
	double _cutoff_freq = 0;
	double _a1 = 0;
	double _a2 = 0;
	double _b0 = 0;
	double _b1 = 0;
	double _b2 = 0;
	Eigen::Vector3d _delay_element_1{0.0, 0.0, 0.0};	// buffered sample -1
	Eigen::Vector3d _delay_element_2{0.0, 0.0, 0.0};	// buffered sample -2
};
#endif
/*------------------------------------------------StatesGroup---------------------------------------------*/
/* Exponent of a Lie algebra */
Eigen::Matrix<double, 3, 3> SO3_Exp(const double &v1, const double &v2, const double &v3)
{
  double &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
  Eigen::Matrix<double, 3, 3> Eye3 = Eigen::Matrix<double, 3, 3>::Identity();
  if (norm > 0.00001)  {
    double r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
    Eigen::Matrix<double, 3, 3> K;
    K << SKEW_SYM_MATRIX(r_ang);
    return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;        /// Roderigous Tranformation
  }
  else  {
    return Eye3;
  }
}
/* Logrithm of a Rotation Matrix */
Eigen::Matrix<double,3,1> SO3_LOG(const Eigen::Matrix<double, 3, 3> &R)
{
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Matrix<double,3,1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

struct StatesGroup
{
public:
  Eigen::Matrix3d rot_end;                                 // [0-2] the estimated attitude
  Eigen::Vector3d pos_end;                                 // [3-5] the estimated position
  Eigen::Vector3d vel_end;                                 // [6-8] the estimated velocity
  Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> cov; // states covariance
  double last_update_time;
  StatesGroup()
  {
    rot_end = Eigen::Matrix3d::Identity();
    pos_end = Eigen::Vector3d::Zero();
    vel_end = Eigen::Vector3d::Zero();
    //cov在so3空间！
    cov = Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity() * 0.0001;//(INIT_COV=0.0001)
    last_update_time = 0;
  }
  ~StatesGroup(){}

  StatesGroup operator+(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)//【SO3+so3=SO3】
  {
    StatesGroup a = *this;
    Eigen::Matrix3d rot_temp;
    rot_temp = SO3_Exp(state_add(0), state_add(1), state_add(2));
    // std::cout<<"========rot_temp========"<<std::endl<< rot_temp <<std::endl;    
    a.rot_end = this->rot_end * rot_temp;
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
    a.cov = this->cov;                          //另外两个operator没有吗？？？cov不变？
    a.last_update_time = this->last_update_time;//另外两个operator没有吗？？？
    return a;
  }
  // StatesGroup &operator+=(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
  // {
  //     this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
  //     this->pos_end += state_add.block<3, 1>(3, 0);
  //     this->vel_end += state_add.block<3, 1>(6, 0);
  //     return *this;
  // }
  Eigen::Matrix<double, DIM_OF_STATES, 1> operator-(const StatesGroup &b)//【SO3相减，差转为so3】
  {
    Eigen::Matrix<double, DIM_OF_STATES, 1> a;
    Eigen::Matrix3d rotd(b.rot_end.transpose() * this->rot_end);
    a.block<3, 1>(0, 0) = SO3_LOG(rotd);
    a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
    a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
    return a;
  }
  static void display(const StatesGroup &state, std::string str = std::string("State: "))
  {
    Eigen::Vector3d angle_axis = SO3_LOG(state.rot_end) * 57.3;
    printf("%s |", str.c_str());
    printf("[%.5f] | ", state.last_update_time);
    printf("(%.3f, %.3f, %.3f) | ", angle_axis(0), angle_axis(1), angle_axis(2));
    printf("(%.3f, %.3f, %.3f) | ", state.pos_end(0), state.pos_end(1), state.pos_end(2));
    printf("(%.3f, %.3f, %.3f) | ", state.vel_end(0), state.vel_end(1), state.vel_end(2));
  }
};

/*------------------------------------------------全局变量---------------------------------------------*/
#ifdef LOW_PASS_FILTER
LowPassFilter2pVector3d lp_filter_acc_{1000, cutoff_freq_};//采样频率，截止频率
LowPassFilter2pVector3d lp_filter_ang_{1000, cutoff_freq_};//采样频率，截止频率
#endif

ros::Subscriber uwb_sub;
ros::Subscriber tag_sub;
ros::Subscriber gnss_gt_sub;
// ros::Subscriber gnss_sub;
// ros::Subscriber gnss_sat_sub;
ros::Subscriber compass_sub;
ros::Subscriber imu_sub;
ros::Publisher pub_odom_filter_local;
nav_msgs::Odometry odom_filter_local;
ros::Publisher pub_odom_filter_global;
nav_msgs::Odometry odom_filter_global;
ros::Publisher pub_odom_reg_local;
nav_msgs::Odometry odom_reg_local;
ros::Publisher pub_odom_reg_global;
nav_msgs::Odometry odom_reg_global;
// ros::Publisher pubPath;
// nav_msgs::Path path;

StatesGroup state_;                                         //fliter local 变量
FILE * m_state_fp;
int star_num_gnss_ = 0;                                     //gps搜星数
bool reset_lio_;                                            //重置lio odometry，使当前时刻lio odom为零
Eigen::Matrix4d m_reset_ = Eigen::Matrix4d::Identity(4,4);  //for local lio odometry reset
double time_gnss_last_ = 0;//数据存储进buffer，时间只记录最后一个(理论上数据也应该只有一个)
double time_compass_last_ = 0;
double time_imu_last_ = 0;
double time_tag_last_ = 0;
double time_uwb_last_ = 0;

sensor_msgs::ImuConstPtr last_imu_;                    //IMU buffer末数据
bool last_imu_flag_ = false;                           //whether there is a last_imu_ data
std::deque<sensor_msgs::Imu::ConstPtr> buf_imu_;       //data buffer for IMU

std::deque<Eigen::Vector3d> buf_pos_gnss_global_;      //gnss数据频率1hz
Eigen::Matrix3d rot_gnss_global_ = Eigen::Matrix3d::Identity();  //just eye

std::deque<Eigen::Matrix3d> buf_rot_compass_global_;   //compass数据频率5hz
std::deque<double> buf_compass_yaw_enu_;               //专门用于计算anchor
Eigen::Vector3d pos_compass_global_ = {0,0,0};                    

std::deque<Eigen::Vector3d> buf_pos_tag_global_;       //tag数据频率5hz
Eigen::Matrix3d rot_tag_global_ = Eigen::Matrix3d::Identity();  //just eye

// std::deque<Eigen::Matrix3d> buf_rot_tag_global_;    //
// std::deque<double> buf_yaw_tag_enu_;                //专门用于计算anchor【弃用tag_rot】tag姿态不准（tag平移离开相机轴心的时候会产生错误姿态值！）并且不稳定（-2度与13度跳变！）

std::deque<Eigen::Vector3d> buf_pos_uwb_global_;       //uwb数据频率5hz
// Eigen::Vector3d pos_uwb_global_;                    //newest uwb global location
Eigen::Matrix3d rot_uwb_global_ = Eigen::Matrix3d::Identity();  //just eye

Eigen::Vector3d pos_filter_global_;                    //newest filter global location
Eigen::Matrix3d rot_filter_global_;                    //newest filter global rotation matrix
double roll_filter_, pitch_filter_, yaw_filter_;       //newest filter global euler angles

Eigen::Vector3d anchor_uwb_blh_;
Eigen::Vector3d anchor_uwb_xyz_;                       //用于计算uwb local到global的转换
Eigen::Matrix3d R_ecef_local_uwb_;                     //uwb local到global的rotation

Eigen::Vector3d anchor_tag0_xyz_;                      //用于计算tag0 local到global的转换
Eigen::Vector3d anchor_tag0_blh_;                      //用于计算tag0 local到global的转换
Eigen::Vector3d anchor_tag1_xyz_;                      //用于计算tag1 local到global的转换
Eigen::Vector3d anchor_tag1_blh_;                      //用于计算tag1 local到global的转换
Eigen::Vector3d anchor_tag2_xyz_;                      //用于计算tag2 local到global的转换
Eigen::Vector3d anchor_tag2_blh_;                      //用于计算tag2 local到global的转换
Eigen::Vector3d anchor_tag3_xyz_;                      //用于计算tag3 local到global的转换
Eigen::Vector3d anchor_tag3_blh_;                      //用于计算tag3 local到global的转换
double yaw_tagbase_enu_;                               //     所有tag local到【当地ENU】的转换 <degree>
Eigen::Matrix3d R_ecef_local_tag_;                     //     所有tag local到global的rotation

Eigen::Vector3d anchor_transition_xyz_;                //用于计算filter local到global的转换
Eigen::Vector3d anchor_transition_blh_;                //用于计算filter local到global的转换
double yaw_transition_enu_;                            //用于计算filter local到【当地ENU】的转换 <degree>
Eigen::Matrix3d R_ecef_local_transition_;              //       filter local到global的rotation

Eigen::Matrix3d R_ecef_enu_;                           //rotation from ENU to ECEF, 从uwb经纬度计算得到
double cost_time_;

#ifdef OUTPUT_FOR_PAPER
FILE* fp_gnss_gt;
FILE* fp_sensor_pos;
FILE* fp_fusion_pos;
#endif
/*------------------------------------------------逻辑控制---------------------------------------------*/
std::mutex mtx_gnss, mtx_imu, mtx_uwb, mtx_tag, mtx_compass;

// bool gnss_ready_ = false;
// bool compass_ready_ = false;
bool transition_ready_ = false;
bool tagbase_ready_ = false;
bool uwbbase_ready_ = false;
// bool tag_ready_ = false;
// bool uwb_ready_ = false;

int filter_flag_ = 0;
int filter_count_ = 0;
int first_imu_ = 1;


/*------------------------------------------------函数---------------------------------------------*/
void dump_filter_state_to_log( FILE *fp )
{
  if (fp != nullptr)
  {
    Eigen::Vector3d rot_angle = Sophus::SO3d( Eigen::Quaterniond( state_.rot_end ) ).log();

    fprintf( fp, "filter count = %i ", filter_count_ ); // Times
    fprintf( fp, "eular angles = %lf %lf %lf ", roll_filter_, pitch_filter_, yaw_filter_);                  // Angle  [1-3]
    fprintf( fp, "pos_end = %lf %lf %lf ", state_.pos_end( 0 ), state_.pos_end( 1 ), state_.pos_end( 2 ) ); // Pos    [4-6]
    fprintf( fp, "vel_end = %lf %lf %lf ", state_.vel_end( 0 ), state_.vel_end( 1 ), state_.vel_end( 2 ) ); // vel    [7-9]
    //filter_flag_: gnss+1,compass+2,tag+4,uwb+8 -> 0.没有观测只有递推 1.只有gnss更新 2.只有compass观测 3.有gnss+compass观测 4.只有tag观测 5.有gnss+tag观测......
    fprintf( fp, "filter flag = %i (gnss+1,compass+2,tag+4,uwb+8)", filter_flag_ ); 
    fprintf( fp, "filter cost time without publish = %lf ", cost_time_); 
    fprintf( fp, "\r\n" );
    fflush( fp );
  }
}

// Avoid abnormal state input //检查速度分量是否过大
bool check_state( StatesGroup &state_inout )
{
  bool is_fail = false;
  for ( int idx = 0; idx < 3; idx++ )
  {
    if ( fabs( state_inout.vel_end(idx) ) > 10 )//任一速度分量>10，认为失败！
    {
      is_fail = true;
      std::cout << "check_state fail !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << state_inout.vel_end.transpose() << std::endl;
      // state_inout.vel_end(idx) = 0.0;
    }
  }
  return is_fail;
}

//imu buffer数据积分，更新状态变量、协方差矩阵、状态变量残差；第一次调用时state_为0；大约消耗0.2ms
void Imu_Process(const std::deque<sensor_msgs::Imu::ConstPtr>& buf_imu, StatesGroup& state_inout, Eigen::MatrixXd d_state_inout)
{
  std::cout<< "Imu_Process......" <<std::endl;
  if(buf_imu.empty()) {
    return;
  }
  auto v_imu = buf_imu;//不改变外部buffer
  if(last_imu_flag_ == true)
  {
    v_imu.push_front(last_imu_);  /*** add the imu of the last frame-tail to the of current frame-head ***///精细！
  }
  if ( check_state( state_inout ) ) //检查速度分量是否过大
  {
    state_inout.display( state_inout, "state_inout" );//打印状态变量
  }
  std::cout<<"state_in"<<std::endl<< state_inout.rot_end <<std::endl<< state_inout.pos_end <<std::endl<< state_inout.vel_end <<std::endl;    

  Eigen::Vector3d acc_imu( 0, 0, 0 ), angvel_avg( 0, 0, 0 ), acc_avg( 0, 0, 0 ), vel_imu( 0, 0, 0 ), pos_imu( 0, 0, 0 );
  Eigen::Matrix3d R_imu( state_inout.rot_end );//Q
  pos_imu = state_inout.pos_end;//P
  vel_imu = state_inout.vel_end;//V
  Eigen::MatrixXd F_x( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Identity() ); //[9x9]递推方程关于状态变量的雅可比
  Eigen::MatrixXd cov_w( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Zero() );   //[9x9]递推方程协方差P
  double dt = 0;

  //IMU数据序列内计算积分、更新协方差矩阵、更新误差状态变量
  for ( std::deque<sensor_msgs::Imu::ConstPtr>::iterator it_imu = v_imu.begin(); it_imu != ( v_imu.end() - 1 ); it_imu++ )
  {
    sensor_msgs::Imu::ConstPtr head = *( it_imu );//it_imu是deque元素的指针，deque元素类型是sensor_msgs::Imu::ConstPtr
    sensor_msgs::Imu::ConstPtr tail = *( it_imu + 1 );
    //平均角速度、加速度
    angvel_avg << 0.5 * ( head->angular_velocity.x + tail->angular_velocity.x ), 0.5 * ( head->angular_velocity.y + tail->angular_velocity.y ),
        0.5 * ( head->angular_velocity.z + tail->angular_velocity.z );
    acc_avg << 0.5 * ( head->linear_acceleration.x + tail->linear_acceleration.x ),
        0.5 * ( head->linear_acceleration.y + tail->linear_acceleration.y ), 0.5 * ( head->linear_acceleration.z + tail->linear_acceleration.z );
    // std::cout<<"DDDDDDDDDDDDDDDangvel_avgDDDDDDDDDDDDDDDDDDD: "<< angvel_avg <<std::endl;
    // std::cout<<"DDDDDDDDDDDDDDDDDDacc_avgDDDDDDDDDDDDDDDD: "<< acc_avg <<std::endl;
    //计算dt
    // if ( tail->header.stamp.toSec() < state_inout.last_update_time ){//不会发生吧？
    //   continue;
    // }
    if ( first_imu_ ){
      first_imu_ = 0;
      dt = 0.05;
    }
    else{
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    // ROS_INFO("last_imu_ 与当前imu时间差dtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdt = %f",dt);
    if ( dt > 0.05 ){//last_imu_与当前imu时间差
      dt = 0.05;
    }
#ifdef LOW_PASS_FILTER
    /* lowpass filter */
    Eigen::Vector3d acc_avg_filtered = lp_filter_acc_.apply(acc_avg);
    Eigen::Vector3d angvel_avg_filtered = lp_filter_ang_.apply(angvel_avg);
    // std::cout<<"DDDDDDDDDDDDDDDDDDacc_avg_filteredDDDDDDDDDDDDDDDD: "<< acc_avg_filtered <<std::endl;
    /* covariance propagation */
    Eigen::Matrix3d acc_avg_skew;
    Eigen::Matrix3d Exp_f = SO3_Exp( angvel_avg_filtered(0)*dt, angvel_avg_filtered(1)*dt, angvel_avg_filtered(2)*dt);
    acc_avg_skew << SKEW_SYM_MATRIX( acc_avg_filtered );
    Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();
    F_x.block< 3, 3 >( 0, 0 ) = Exp_f.transpose();                                // Q关于Q
    F_x.block< 3, 3 >( 3, 3 ) = Eigen::Matrix3d::Identity();                      // P关于P
    F_x.block< 3, 3 >( 3, 6 ) = Eigen::Matrix3d::Identity() * dt;                 // P关于V
    F_x.block< 3, 3 >( 6, 0 ) = -R_imu * acc_avg_skew * dt;                       // V关于Q
    Eigen::Matrix3d cov_acc_diag, cov_gyr_diag, cov_omega_diag;
    cov_omega_diag = Eigen::Vector3d( COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG ).asDiagonal();//(0.1,0.1,0.1) 【固定值】【角速度误差=陀螺仪误差】
    cov_gyr_diag = Eigen::Vector3d( COV_VEL_NOISE_DIAG, COV_VEL_NOISE_DIAG, COV_VEL_NOISE_DIAG ).asDiagonal();//(0.2，0.2，0.2)【固定值】【速度误差】
    cov_acc_diag = Eigen::Vector3d( COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG ).asDiagonal();//(0.4,0.4,0.4)【固定值】【加速度误差】
    //依次是Q、P、V噪声
    cov_w.block<3,3>(0,0) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;//(角速度误差*dt)^2=角度噪声，ESKF小册子里没有姿态修正，但是武大KF-GINS乘了姿态
    cov_w.block<3,3>(3,3) = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt;  //(速度误差*姿态修正*dt)^2=位置噪声，ESKF小册子里没有这一项！！！武大KF-GINS也没有
    cov_w.block<3,3>(6,6) = cov_acc_diag * dt * dt;                              //(加速度误差*dt)^2=速度噪声，ESKF小册子里没有姿态修正，但是武大KF-GINS乘了姿态
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;//协方差递推，噪声累积；[9x9]*[9x9]*[9x9] + [9x9]
    /* error state propagation */
    d_state_inout = F_x * d_state_inout;
    //状态x递推
    R_imu = R_imu * Exp_f; // R * w^ * dt
    acc_imu = R_imu * acc_avg_filtered + Eigen::Vector3d(0.0, 0.0, 9.805); // R * a - g
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    vel_imu = vel_imu + acc_imu * dt;
#else
    /* covariance propagation */
    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f = SO3_Exp( angvel_avg(0)*dt, angvel_avg(1)*dt, angvel_avg(2)*dt);
    acc_avr_skew << SKEW_SYM_MATRIX( acc_avg );
    Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();
    F_x.block< 3, 3 >( 0, 0 ) = Exp_f.transpose();                                // Q关于Q
    F_x.block< 3, 3 >( 3, 3 ) = Eigen::Matrix3d::Identity();                      // P关于P
    F_x.block< 3, 3 >( 3, 6 ) = Eigen::Matrix3d::Identity() * dt;                 // P关于V
    F_x.block< 3, 3 >( 6, 0 ) = -R_imu * acc_avr_skew * dt;                       // V关于Q
    Eigen::Matrix3d cov_acc_diag, cov_gyr_diag, cov_omega_diag;
    cov_omega_diag = Eigen::Vector3d( COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG ).asDiagonal();//(0.1,0.1,0.1) 【固定值】【角速度误差=陀螺仪误差】
    cov_gyr_diag = Eigen::Vector3d( COV_VEL_NOISE_DIAG, COV_VEL_NOISE_DIAG, COV_VEL_NOISE_DIAG ).asDiagonal();//(0.2，0.2，0.2)【固定值】【速度误差】
    cov_acc_diag = Eigen::Vector3d( COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG ).asDiagonal();//(0.4,0.4,0.4)【固定值】【加速度误差】
    //依次是Q、P、V噪声
    cov_w.block< 3, 3 >( 0, 0 ) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;//(角速度误差*dt)^2=角度噪声，ESKF小册子里没有姿态修正，但是武大KF-GINS乘了姿态
    cov_w.block< 3, 3 >( 3, 3 ) = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt;  //(速度误差*姿态修正*dt)^2=位置噪声，ESKF小册子里没有这一项！！！武大KF-GINS也没有
    cov_w.block< 3, 3 >( 6, 6 ) = cov_acc_diag * dt * dt;                              //(加速度误差*dt)^2=速度噪声，ESKF小册子里没有姿态修正，但是武大KF-GINS乘了姿态
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;//协方差递推，噪声累积；[9x9]*[9x9]*[9x9] + [9x9]
    /* error state propagation */
    d_state_inout = F_x * d_state_inout;
    //状态x递推
    R_imu = R_imu * Exp_f; // R * w^ * dt
    acc_imu = R_imu * acc_avg + Eigen::Vector3d(0.0, 0.0, 9.805); // R * a - g
    // acc_imu = R_imu * acc_avg;
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    vel_imu = vel_imu + acc_imu * dt;
#endif
  }

  state_inout.last_update_time = v_imu.back()->header.stamp.toSec();
  state_inout.vel_end = vel_imu;
  state_inout.rot_end = R_imu;
  state_inout.pos_end = pos_imu;
  last_imu_ = v_imu.back();//deque: front\back是引用 begin\end是指针
  last_imu_flag_ = true;
  std::cout<<"state_out"<<std::endl<< state_inout.rot_end <<std::endl<< state_inout.pos_end <<std::endl<< state_inout.vel_end <<std::endl;    
}

Eigen::Vector3d ecef_blh2xyz(Eigen::Vector3d& BLH)//wgs84标准下经Breite纬Lange高Hohe转xyz  https://blog.csdn.net/u010384390/article/details/78532775
{
  double a = 6378137;
  double f = 1 / 298.257223563;
  double b = a * (1 - f);//短半轴
  // std::cout<<"b:" << b <<std::endl;
  double e2 = (a*a - b*b) / (a*a);//第一偏心率
  // std::cout<<"e2:" << e2 <<std::endl;
  // double NN =  1 - e2 * sin(BLH(0) * D2R) * sin(BLH(0) * D2R);
  double N = a / sqrt(1 - e2 * sin(BLH(0) * D2R) * sin(BLH(0) * D2R));
  // std::cout<<"N:" << N <<std::endl;
  double WGS84_X = (N + BLH(2)) * cos(BLH(0) * D2R) * cos(BLH(1) * D2R);
  double WGS84_Y = (N + BLH(2)) * cos(BLH(0) * D2R) * sin(BLH(1) * D2R);
  double WGS84_Z = (N * (1 - e2) + BLH(2)) * sin(BLH(0) * D2R);
  Eigen::Vector3d xyz;
  xyz << WGS84_X, WGS84_Y, WGS84_Z;
  return xyz;
}

Eigen::Vector3d ecef_xyz2blh(Eigen::Vector3d& xyz)//wgs84标准下xyz转经Breite纬Lange高Hohe  https://zhidao.baidu.com/question/429795836325466052.html
{
  double a = 6378137;
  double f = 1/298.257223563;
  double b = a * (1 - f);//短半轴
  double e2 = (a * a - b * b) / (a*a);//第一偏心率
  double eq2 = (a * a - b * b) / (b*b);//
  double theta = atan((xyz(2)*a) / (sqrt(xyz(0)*xyz(0)+xyz(1)*xyz(1))*b));
  double WGS84_L = R2D * atan(xyz(1) / xyz(0));
  double WGS84_B = R2D * atan((xyz(2) +  eq2*b*sin(theta)*sin(theta)*sin(theta)) / (sqrt(xyz(0)*xyz(0)+xyz(1)*xyz(1)) - e2*a*cos(theta)*cos(theta)*cos(theta)));
  double N = a / sqrt(1 - e2 * sin(WGS84_B * D2R) * sin(WGS84_B * D2R));
  // std::cout<<"N:" << N <<std::endl;
  double WGS84_H = (sqrt(xyz(0)*xyz(0)+xyz(1)*xyz(1))/cos(WGS84_B*D2R)) - N;
  Eigen::Vector3d blh;
  blh << (180 - WGS84_B), WGS84_L, WGS84_H;//修正到东半球！！！
  return blh;
}

void initializel_tag(const std::string tag_file)//tag_file没有后缀名
{
  //从与tagfile同名的.dat文件读取tag原点的ECEF系xyz坐标和ENU系yaw
  std::string tag_anchor = tag_file + ".dat";
  std::ifstream infile;
  infile.open(tag_anchor.data());//将文件流对象和文件连接起来
  assert(infile.is_open());
  double d1,d2,d3;
  infile >> d1;//【赞美斌哥】
  infile >> d2;
  infile >> d3;
  anchor_tag0_blh_ << d1,d2,d3;
  infile >> yaw_tagbase_enu_;

  infile >> d1;//【赞美斌哥】
  infile >> d2;
  infile >> d3;
  anchor_tag1_blh_ << d1,d2,d3;

  infile >> d1;//【赞美斌哥】
  infile >> d2;
  infile >> d3;
  anchor_tag2_blh_ << d1,d2,d3;

  infile >> d1;//【赞美斌哥】
  infile >> d2;
  infile >> d3;
  anchor_tag3_blh_ << d1,d2,d3;
  infile.close();

  anchor_tag0_xyz_ = ecef_blh2xyz(anchor_tag0_blh_);
  anchor_tag1_xyz_ = ecef_blh2xyz(anchor_tag1_blh_);
  anchor_tag2_xyz_ = ecef_blh2xyz(anchor_tag2_blh_);
  anchor_tag3_xyz_ = ecef_blh2xyz(anchor_tag3_blh_);
  std::cout<<std::endl<<"tag0's anchor in BLH = "<<std::endl<<anchor_tag0_blh_;
  std::cout<<std::endl<<"tag0's anchor in xyz = "<<std::endl<<anchor_tag0_xyz_;
  std::cout<<std::endl<<"tags' yaw in degree = " <<yaw_tagbase_enu_;
  std::cout<<std::endl<<"tag1's anchor in BLH = "<<std::endl<<anchor_tag1_blh_;
  std::cout<<std::endl<<"tag1's anchor in xyz = "<<std::endl<<anchor_tag1_xyz_;
  std::cout<<std::endl<<"tag2's anchor in BLH = "<<std::endl<<anchor_tag2_blh_;
  std::cout<<std::endl<<"tag2's anchor in xyz = "<<std::endl<<anchor_tag2_xyz_;
  std::cout<<std::endl<<"tag3's anchor in BLH = "<<std::endl<<anchor_tag3_blh_;
  std::cout<<std::endl<<"tag3's anchor in xyz = "<<std::endl<<anchor_tag3_xyz_;

  //anchor确定了ENU坐标和ECEF坐标的转换关系，tag的 R_ecef_enu_ 近似使用uwb计算得到的
  double sin_yaw_diff = std::sin(yaw_tagbase_enu_*D2R);//从local到enu：sin(yaw);从enu到local：sin(-yaw)!!!!!!!!!!!!!!!!!!!!!!!!!!
  double cos_yaw_diff = std::cos(yaw_tagbase_enu_*D2R);
  Eigen::Matrix3d R_enu_local;
  R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                 sin_yaw_diff,  cos_yaw_diff, 0,
                 0           ,  0           , 1;
  R_ecef_local_tag_ = R_ecef_enu_ * R_enu_local;
  tagbase_ready_ = true;
  std::cout<<std::endl<<"Tags are ready!!!" <<std::endl;
}

void initializel_uwb(const std::string uwb_file)//uwb_file没有后缀名
{
  //从与uwb_file同名的.dat文件读取uwb原点的ECEF系xyz坐标和ENU系yaw
  std::string uwb_anchor = uwb_file + ".dat";
  std::ifstream infile;
  infile.open(uwb_anchor.data());//将文件流对象和文件连接起来
  assert(infile.is_open());
  double d1,d2,d3;
  infile >> d1;//【赞美斌哥】
  infile >> d2;
  infile >> d3;
  anchor_uwb_blh_ << d1,d2,d3;
  double yaw_uwb_enu;
  infile >> yaw_uwb_enu;
  infile.close();

  anchor_uwb_xyz_ = ecef_blh2xyz(anchor_uwb_blh_);
  std::cout<<std::endl<<"uwb's anchor in BLH = "<<std::endl<<anchor_uwb_blh_;
  std::cout<<std::endl<<"uwb's anchor in xyz = "<<std::endl<<anchor_uwb_xyz_;
  std::cout<<std::endl<<"uwb' yaw in degree = " <<yaw_uwb_enu;

  //anchor确定了ENU坐标和ECEF坐标的转换关系
  double lat = anchor_uwb_blh_.x()*D2R, lon = anchor_uwb_blh_.y()*D2R;//经纬
  double sin_lat = sin(lat), cos_lat = cos(lat);
  double sin_lon = sin(lon), cos_lon = cos(lon);
  R_ecef_enu_ << -sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon,
                  cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon,
                  0      ,  cos_lat        , sin_lat;
  double sin_yaw_diff = std::sin(yaw_uwb_enu*D2R);//从local到enu：sin(yaw);从enu到local：sin(-yaw)!!!!!!!!!!!!!!!!!!!!!!!!!!
  double cos_yaw_diff = std::cos(yaw_uwb_enu*D2R);
  Eigen::Matrix3d R_enu_local;
  R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                 sin_yaw_diff,  cos_yaw_diff, 0,
                 0           ,  0           , 1;
  R_ecef_local_uwb_ = R_ecef_enu_ * R_enu_local;
  uwbbase_ready_ = true;
  std::cout<<std::endl<<"uwb are ready!!!" <<std::endl;
}

bool isIdentity(const Eigen::MatrixXd& mat, double precision = 1e-7)
{
    return mat.isApprox(Eigen::MatrixXd::Identity(mat.rows(), mat.cols()), precision);
}

void filter_global2local(const Eigen::Vector3d& pos_global, const Eigen::Matrix3d& rot_global, StatesGroup& g_state, Eigen::VectorXd& state)
{
  if(!transition_ready_)
  {
    ROS_INFO("transition not ready, need anchor_transition to make filter_global2local work!");
  }
  // std::cout<<std::endl<<"======== R_ecef_local_transition_ ========"<<std::endl<< R_ecef_local_transition_ <<std::endl;  
  // std::cout<<std::endl<<"======== anchor_transition_xyz_ ========"<<std::endl<< anchor_transition_xyz_ <<std::endl;  
  // std::cout<<std::endl<<"======== pos_global ========"<<std::endl<< pos_global <<std::endl;  
  Eigen::Vector3d pos_local;
  pos_local = R_ecef_local_transition_.inverse() * (pos_global - anchor_transition_xyz_);//基准点使用 anchor_transition_xyz_

  Eigen::Matrix3d rot_local;
  rot_local = R_ecef_local_transition_.inverse() * rot_global;
  std::cout<<std::endl<<"======== measurement ========"<<std::endl<< rot_local <<std::endl<< pos_local <<std::endl;  
  // Eigen::Matrix3d rot_diff =  rot_local.inverse()*g_state.rot_end;
  // std::cout<<std::endl<<"======== rot_diff ========"<<std::endl<< rot_diff <<std::endl;  
  state.head<3>() = SO3_LOG(g_state.rot_end.inverse()*rot_local);//姿态在先！！！！！！！！！！！！！！！！
  state.tail<3>() = pos_local - g_state.pos_end;
  if(isIdentity(rot_global))//观测量只有pos没有rot的情况
  {
    state.head<3>() << 0,0,0;
    std::cout<<"ONLY position" <<std::endl;    

#ifdef OUTPUT_FOR_PAPER
  std::string bufferfile = std::to_string(state_.last_update_time) + "," + std::to_string(pos_local(0))+ "," 
                         + std::to_string(pos_local(1))+ "," + std::to_string(pos_local(2)) + "\n";
  const char* buffer_file = bufferfile.c_str();
  std::cout<<"sensor buffer_file: "<< buffer_file <<std::endl;
  fwrite(buffer_file, 1, 50, fp_sensor_pos);
#endif


  }
  double pp = sqrt(pos_global(0)*pos_global(0) + pos_global(1)*pos_global(1) + pos_global(2)*pos_global(2));
  if(pp < 1e-7)////观测量只有rot没有pos的情况
  {
    state.tail<3>() << 0,0,0;
    std::cout<<"ONLY rotation" <<std::endl;    
  }
  // std::cout<<"========  g_state   ========"<<std::endl<< g_state <<std::endl;    
  // std::cout<<"======== meas_vec = measurement-g_state ========"<<std::endl<< state <<std::endl;    
}

//IErKF局部坐标系state（pos+rot_mat），转到全局pos+rot
void filter_local2global(const  StatesGroup& state, Eigen::Vector3d& pos_global, Eigen::Matrix3d& rot_global)
{
  if(!transition_ready_)
  {
    ROS_INFO("transition not ready, need anchor_transition to make filter_local2global work!");
  }
  pos_global = R_ecef_local_transition_ * state.pos_end + anchor_transition_xyz_;       //基准点使用 anchor_transition_xyz_
  rot_global = R_ecef_local_transition_ * state.rot_end;
}

//Tag局部位姿转为全局位姿，其中全局位姿需要rot_mat（用于ESKF）和yaw_enu（用于初始计算anchor）两种形式
void tag_local2global(const Eigen::Vector3d& pos_local, const Eigen::Matrix3d& rot_local, const int tag_id, 
                      Eigen::Vector3d& pos_global, Eigen::Matrix3d& rot_global, double& yaw_enu)
{
  // if(!tagbase_ready_)
  // {
  //   ROS_INFO("tagbase not ready, need anchor_tag to make tag_local2global work!");
  // }
  switch(tag_id)
  {
    case 0:
      pos_global = R_ecef_local_tag_ * pos_local + anchor_tag0_xyz_;       //基准点使用 anchor_tag_xyz_
      rot_global = R_ecef_local_tag_ * rot_local;
      break;
    case 1:
      pos_global = R_ecef_local_tag_ * pos_local + anchor_tag1_xyz_;       //基准点使用 anchor_tag_xyz_
      rot_global = R_ecef_local_tag_ * rot_local;
      break;
    case 2:
      pos_global = R_ecef_local_tag_ * pos_local + anchor_tag2_xyz_;       //基准点使用 anchor_tag_xyz_
      rot_global = R_ecef_local_tag_ * rot_local;
      break;
    case 3:
      pos_global = R_ecef_local_tag_ * pos_local + anchor_tag3_xyz_;       //基准点使用 anchor_tag_xyz_
      rot_global = R_ecef_local_tag_ * rot_local;
      break;
    case 7:
      pos_global = R_ecef_local_tag_ * pos_local + anchor_tag0_xyz_;       //室内测试用小tag
      rot_global = R_ecef_local_tag_ * rot_local;
      break;      
    default:
      std::cout<< "Wrong Tag ID!!!!!!"<<std::endl;
      break;
  }
  //rotation to yaw_local to yaw_enu
  double yaw_local;
  double sy = sqrt(rot_local(0,0)*rot_local(0,0) + rot_local(1,0)*rot_local(1,0));//rot_local.at<double>(0,0)
  bool singular = sy < 1e-6;
  if(!singular) {
    yaw_local = atan2(rot_local(1, 0), rot_local(0, 0)) * R2D; 
  }
  else  {    
    yaw_local = 0;
  }
  // std::cout<< "!!!!!!!!!!!!!!!!!!!!!!rot_local = "<< std::endl<< rot_local <<std::endl;
  // std::cout<< "!!!!!!!!!!!!!!!!!!!!!!yaw_local = "<< yaw_local <<std::endl;
  yaw_enu = yaw_local + yaw_tagbase_enu_;
}

//uwb局部位置转为全局位置
void uwb_local2global(const Eigen::Vector3d& pos_local, Eigen::Vector3d& pos_global)
{
    pos_global = R_ecef_local_uwb_ * pos_local + anchor_uwb_xyz_; 
}

void uwb_callback(const nlink_parser::LinktrackNodeframe2& uwbs)
{
  std::cout<< "uwb_callback......" <<std::endl;
  Eigen::Vector3d pos_local, pos_uwb_global;
  // pos_local << uwbs.pos_3d[0], uwbs.pos_3d[1], -uwbs.pos_3d[2];//Anchor全部位于一个平面时候，UWB系统无法区分Tag位于平面上还是下，默认取负值!
  pos_local << uwbs.pos_3d[0]+0.3, uwbs.pos_3d[1]+0.25, -uwbs.pos_3d[2];//nodeframe2_0在右!!
  uwb_local2global(pos_local, pos_uwb_global);
  time_uwb_last_ = uwbs.header.stamp.toSec();//LinktrackNodeframe2没有header！添加
  mtx_uwb.lock();
  buf_pos_uwb_global_.push_back(pos_uwb_global);
  mtx_uwb.unlock();
}

void tag_callback(const apriltag_ros::AprilTagDetectionArray& tags) //callback still run even if there is no detections!
{
  std::cout<< "tag_callback......" <<std::endl;
  int size = tags.detections.size();//多个tag时id顺序不稳定，同时出现哪个更准？取平均！
  if(size == 0)
  {
    return;
  }
  int id;
  Eigen::Vector3d pos_tag_local;
  Eigen::Affine3d affine;
  Eigen::Matrix3d rot_tag_local;
  Eigen::Matrix4d trans_cam2tag;//tag在相机系下位姿，也是tag系坐标转换到相机系的T
  Eigen::Matrix4d trans_tag2cam;//相机在tag系下位姿
  Eigen::Matrix4d trans_local2tag;//tag在tag-local坐标系下位姿
  Eigen::Matrix4d trans_local2imu;//imu在tag-local坐标系下位姿
  Eigen::Matrix4d trans_cam2imu;//imu在cam坐标系的外参
  trans_local2tag << 0,-1, 0, 0,
                    -1, 0, 0, 0,
                     0, 0,-1, 0,
                     0, 0, 0, 1;//Tag头(-Y)朝前平放，【简化认为tag坐标系与cam坐标系平行，即为xyz:右下前】
  trans_cam2imu << 0,-1, 0, 0,
                  -1, 0, 0, 0,
                   0, 0,-1, 0,
                   0, 0, 0, 1;//相机向下放置时
  // trans_local2tag << 0, 0, 1, 0,
  //                   -1, 0, 0, 0,
  //                    0,-1, 0, 0,
  //                    0, 0, 0, 1;//Tag背(Z)朝前竖放
  // trans_cam2imu << 0,-1, 0, 0,
  //                  0, 0,-1, 0,
  //                  1, 0, 0, 0,
  //                  0, 0, 0, 1;//相机向前放置时

  // /////////////////////////////////////基于yaw的多个tag求平均，如何得到平均rot_global呢？
  /////////////////////////////////////只用第一个tag，不求平均!!!
  id = tags.detections[0].id[0];

  geometry_msgs::Pose pose_temp;
  pose_temp.position = tags.detections[0].pose.pose.pose.position;
  // pose_temp.orientation.x = tags.detections[0].pose.pose.pose.orientation.x;// tag orientation bad! use compass orientation!
  // pose_temp.orientation.y = tags.detections[0].pose.pose.pose.orientation.y;
  // pose_temp.orientation.z = tags.detections[0].pose.pose.pose.orientation.z;
  // pose_temp.orientation.w = tags.detections[0].pose.pose.pose.orientation.w;
  // tf::poseMsgToEigen(pose_temp, affine);//
  // trans_cam2tag = affine.matrix();
  Eigen::Matrix3d rot_compass_global;
  if(!buf_rot_compass_global_.empty())//有新的compass观测
  {
    rot_compass_global = buf_rot_compass_global_.front();
  }
  else
  {
    printf("aaaaaaaaaaaaaaaaaaaaaaaaa");
    return;// no valid tag position without compass pose!
  }
  Eigen::Matrix3d rot_cam2tag;//【简化认为tag坐标系与cam坐标系平行，即为xyz:右下前】
  //tag相对cam姿态 = imu相对cam姿态*imu全局姿态.inverse()*(local全局姿态*tag相对local姿态)
  rot_cam2tag = trans_cam2imu.block<3,3>(0,0)*rot_compass_global.inverse()*(R_ecef_local_tag_*trans_local2tag.block<3,3>(0,0));
  trans_cam2tag.block<3,3>(0,0) = rot_cam2tag;
  trans_cam2tag.block<1,3>(3,0) = Eigen::Vector3d::Zero().transpose();
  trans_cam2tag(0,3) = pose_temp.position.x;
  trans_cam2tag(1,3) = pose_temp.position.y;
  trans_cam2tag(2,3) = pose_temp.position.z;
  trans_cam2tag(3,3) = 1;
  trans_tag2cam = trans_cam2tag.inverse();

  trans_local2imu = trans_local2tag * trans_tag2cam * trans_cam2imu;
  // std::cout<< "trans_local2imu:" <<std::endl<< trans_local2imu <<std::endl;

  rot_tag_local = trans_local2imu.block<3,3>(0,0);
  pos_tag_local << trans_local2imu(0,3),trans_local2imu(1,3),trans_local2imu(2,3);
  // printf("pos_cam2tag ================================== %lf %lf %lf \n", pose_temp.position.x, pose_temp.position.y, pose_temp.position.z);
  // printf("local_tag_ position  ========================= %lf %lf %lf \n", pos_tag_local(0), pos_tag_local(1), pos_tag_local(2));
  double sy = sqrt(rot_tag_local(0,0)*rot_tag_local(0,0) + rot_tag_local(1,0)*rot_tag_local(1,0));
  double roll_tag = atan2(rot_tag_local(2, 1), rot_tag_local(2, 2)) * R2D;
  double pitch_tag = atan2(-rot_tag_local(2, 0), sy) * R2D;
  double yaw_tag = atan2(rot_tag_local(1, 0), rot_tag_local(0, 0)) * R2D; 
  // printf("local_tag_ eular angles ====================== %lf %lf %lf \n", roll_tag, pitch_tag, yaw_tag);

  Eigen::Vector3d pos_tag_global;
  Eigen::Matrix3d rot_tag_global; 
  double yaw_tag_enu; 
  tag_local2global(pos_tag_local, rot_tag_local, id, pos_tag_global, rot_tag_global, yaw_tag_enu);

  time_tag_last_ = tags.header.stamp.toSec();
  // std::unique_lock<std::mutex> lock(mtx_tag);
  // std::lock_guard<std::mutex> lock(mtx_tag);
  mtx_tag.lock();
  buf_pos_tag_global_.push_back(pos_tag_global);
  // buf_rot_tag_global_.push_back(rot_tag_global);
  // buf_yaw_tag_enu_.push_back(yaw_tag_enu);
  mtx_tag.unlock();
}

#ifdef OUTPUT_FOR_PAPER
void gnss_gt_callback(const sensor_msgs::NavSatFix& data)//
{
  // std::cout<<"gnss_gt data callback~~~~~~ ~~~~~~" <<std::endl;
  Eigen::Vector3d blh;
  blh << data.longitude, data.latitude, data.altitude;
  Eigen::Vector3d pos_global = ecef_blh2xyz(blh);                   //ECEF三坐标
  Eigen::Vector3d pos_local = R_ecef_local_transition_.inverse() * (pos_global - anchor_transition_xyz_);//基准点使用 anchor_transition_xyz_

  std::string bufferfile = std::to_string(data.header.stamp.toSec()) + "," + std::to_string(pos_local(0))+ "," 
                         + std::to_string(pos_local(1))+ "," + std::to_string(pos_local(2)) + "\n";
  const char* buffer_file = bufferfile.c_str();
  // std::cout<<"gnss_gt buffer_file: "<< buffer_file <<std::endl;
  fwrite(buffer_file, 1, 50, fp_gnss_gt);

}
#endif

void gnss_callback(const sensor_msgs::NavSatFixConstPtr& data_gnss, const fdilink_ahrs::satelliteConstPtr& data_sat)//
{
  std::cout<< "gnss_callback......" <<std::endl;
  Eigen::Vector3d blh;
  blh << data_gnss->longitude, data_gnss->latitude, data_gnss->altitude;
  std::cout << std::setprecision(9) << blh <<std::endl;    
  Eigen::Vector3d pos_gnss_global = ecef_blh2xyz(blh);                   //ECEF三坐标

  mtx_gnss.lock(); // 写独占锁，等待读锁全部释放
  buf_pos_gnss_global_.push_back(pos_gnss_global);
  mtx_gnss.unlock();

  time_gnss_last_ = data_gnss->header.stamp.toSec();
  star_num_gnss_ = data_sat->num_satellites;
  std::cout<< star_num_gnss_ <<std::endl;    
}

void compass_callback(const fdilink_ahrs::compassConstPtr& data_mag)//
{
  std::cout<< "compass_callback......" <<std::endl;

  double compass_roll, compass_pitch, compass_yaw;
  Eigen::Vector3d pos_gnss_global;
  Eigen::Matrix3d rot_compass_global;

  compass_roll = 3.1415 - data_mag->roll;   //<rad> fdilink 传感器倒放!
  compass_pitch = -data_mag->pitch;   //<rad>
  compass_yaw = - data_mag->heading;   //<rad>                                                                 【基准是真北】
  printf("eular angles = %lf %lf %lf \n", compass_roll*R2D, compass_pitch*R2D, compass_yaw*R2D);                  // Angle  [1-3]

  // double sin_yaw_diff = std::sin(compass_yaw);//从local到enu：sin(yaw);从enu到local：sin(-yaw)!!!!!!!!!!!!!!!!!!!!!!!!!!
  // double cos_yaw_diff = std::cos(compass_yaw);
  Eigen::Matrix3d rot_gnss_enu;
  // rot_gnss_enu << cos_yaw_diff, -sin_yaw_diff, 0,
  //                 sin_yaw_diff,  cos_yaw_diff, 0,
  //                 0           ,  0           , 1;
  rot_gnss_enu = Eigen::AngleAxisd(compass_yaw, Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(compass_pitch, Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(compass_roll, Eigen::Vector3d::UnitX());
  rot_compass_global = R_ecef_enu_*rot_gnss_enu;         //R_ecef_enu_ 是近似使用map计算得到的

  mtx_compass.lock(); // 写独占锁，等待读锁全部释放
  buf_rot_compass_global_.push_back(rot_compass_global);
  buf_compass_yaw_enu_.push_back(compass_yaw);
  mtx_compass.unlock();

  time_compass_last_ = data_mag->header.stamp.toSec();
}

void imu_callback(const sensor_msgs::ImuPtr &imu_msg)
{
  if(!transition_ready_)
  {
    return;//transition not ready, need no IMU data!
  }
  time_imu_last_ = imu_msg->header.stamp.toSec();

  //for test
  // imu_msg->angular_velocity.x = 0;
  // imu_msg->angular_velocity.y = 0;
  // imu_msg->angular_velocity.z = 0;
  // imu_msg->linear_acceleration.x = 0;
  // imu_msg->linear_acceleration.y = 0;
  // imu_msg->linear_acceleration.z = 0;

  mtx_imu.lock();
  buf_imu_.push_back(imu_msg);
  mtx_imu.unlock();
}

void filtering_process()
{
  ROS_INFO("////////////////////////MSN filter start//////////////////////////");
  ros::Rate rate_filter(1);//filter频率应>=传感器频率（IMU除外）！拿到新的数据立刻使用不要等待！
  // 用于传递协方差用的矩阵, 用于计算kalman增益用的, 单位阵
  Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> I_STATE;
  I_STATE.setIdentity();
  Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> I_KH;
  //误差状态变量
  Eigen::Matrix<double, DIM_OF_STATES, 1> d_state;
  d_state.setZero();
  //设置状态变量初始协方差矩阵
  state_.cov.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * R_init_COV;   //   1e-5
  state_.cov.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity() * P_init_COV;   //   1e-5
  state_.cov.block(6, 6, 3, 3) = Eigen::Matrix3d::Identity() * V_init_COV;   //   1e-5
  while(ros::ok())
  {
    double time_start = ros::Time::now().toSec();
    // double imu_start = ros::Time::now().toSec();
    mtx_imu.lock();//锁定的时候imu_callback数据会等待吗？？？？当前进程耗时多少？？？？
    Imu_Process(buf_imu_, state_, d_state);//imu buffer数据积分，更新状态变量、协方差矩阵、状态变量残差；第一次调用时state_为0；大约消耗0.2ms
    buf_imu_.clear();//buf_imu_核减
    mtx_imu.unlock();
    // double imu_end = ros::Time::now().toSec();
    // double imu_time = imu_end - imu_start;
    // std::cout<<"========IMU processing time cost========  "<< imu_time <<std::endl;    
    // std::cout<<"======== state_.cov ========"<<std::endl<< state_.cov <<std::endl;    

    if(!buf_pos_gnss_global_.empty() && star_num_gnss_ > threshold_gnss_star_ && mtx_gnss.try_lock())//gnss buffer没被占用 && 有新的gnss观测 && 搜星数>阈值
    {
      ROS_INFO("!!!!!!!!NEW GNSS!!!!!!!!");
      /***Measuremnt Jacobian matrix H and measurents vector ***/
      state_.last_update_time = time_gnss_last_;
      Eigen::VectorXd meas_vec(6);
      Eigen::Vector3d pos_gnss_global = buf_pos_gnss_global_.front();
      filter_global2local(pos_gnss_global, rot_gnss_global_, state_, meas_vec);//meas_vec =（状态变量-观测量）转换到局部坐标系的差值，姿态三个自由度 + pos三个自由度
      buf_pos_gnss_global_.clear();//理论上buffer内只有一个元素，如果正常情况下会存储多个元素，那么此处观测量应该更新buf_pos_gnss_global_.size()次！
      mtx_gnss.unlock(); // 释放读锁

      std::cout<<"======== state_ before new GNSS========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
      std::cout<<"========  meas_vec  ========"<<std::endl<< meas_vec <<std::endl;    
      // Eigen::Matrix< double, DIM_OF_STATES, 1 > d_state;//[9x1]
      d_state.setZero();      // 误差状态反馈到系统状态后,将误差状态清零
      Eigen::MatrixXd Hsub(6,9);
      Hsub.setZero();
      //6维观测（姿态的观测与姿态的状态变量都是3维李代数），姿态观测rho对状态变量rho的Jacobian是1，pos观测yxz对pos_end的Jacobian是1，观测对状态变量vel_end的Jacobian是0。
      Hsub << 1,0,0,0,0,0,0,0,0,
              0,1,0,0,0,0,0,0,0,
              0,0,1,0,0,0,0,0,0,
              0,0,0,1,0,0,0,0,0,
              0,0,0,0,1,0,0,0,0,
              0,0,0,0,0,1,0,0,0;          
      // construct measurement noise matrix
      Eigen::MatrixXd R_gnss(6,6);
      // R_gnss = gnssdata.std.cwiseProduct(gnssdata.std).asDiagonal();
      R_gnss<<GNSS_OBSERVE_COV,0,0,0,0,0,
              0,GNSS_OBSERVE_COV,0,0,0,0,
              0,0,GNSS_OBSERVE_COV,0,0,0,
              0,0,0,GNSS_OBSERVE_COV,0,0,
              0,0,0,0,GNSS_OBSERVE_COV,0,
              0,0,0,0,0,GNSS_OBSERVE_COV; 

      /*** Error State Kalman Filter Update ***/
      auto temp         = Hsub * state_.cov * Hsub.transpose() + R_gnss;//[6x6]=[6x9]*[9x9]*[9x6]+[6x6]
      Eigen::MatrixXd K = state_.cov * Hsub.transpose() * temp.inverse();//[9x6]=[9x9]*[9x6]*[6x6]
      // std::cout<<"========  temp  ========"<<std::endl<< temp <<std::endl;    
      // std::cout<<"========  K  ========"<<std::endl<< K <<std::endl;    
      //【Jacobian(Hsub)、g_state_.cov、K、solution都是so3相关变量】
      // ESKF(ErKF)：meas_vec是观测量差值，d_state_是状态变量差值，ESKF小册子给的是状态变量和观测量nominal值，但是残差值更合理，武大KF-GINS也是残差值
      // 如果每次观测量更新后都进行误差状态反馈，则Imu_Process不会影响d_state，它的值直到下一次观测量更新之前都是0，下式可以简化为：d_state = K * meas_vec
      d_state = K * meas_vec - K * Hsub * d_state;//[9x1] = [9x6]*[6x1]-[9x6]*[6x9]*[9x1]
      // std::cout<<"========d_state========"<<std::endl<< d_state <<std::endl;    
      // Eigen::Matrix3d rot_temp = SO3_Exp(d_state(0), d_state(1), d_state(2));
      // std::cout<<"========rot_temp========"<<std::endl<< rot_temp <<std::endl; 
      state_ = state_ + d_state;//【SO3+so3=SO3】

      /*** Covariance Update ***/
      I_KH = I_STATE - K * Hsub;//[9x9] = [9x9]-[9x6]*[6x9]
      state_.cov = I_KH * state_.cov * I_KH.transpose() + K * R_gnss * K.transpose();//[9x9] = [9x9]*[9x9]*[9x9] + [9x6]*[6x6]*[6x9]
      // std::cout<<"======== state_.cov ========"<<std::endl<< state_.cov <<std::endl;    

      filter_flag_ += 1;
      std::cout<<"======== state_ after gnss filter ========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
    }

    if(!buf_rot_compass_global_.empty() && mtx_compass.try_lock())//compass buffer没被占用 && 有新的compass观测
    {
      ROS_INFO("!!!!!!!!NEW compass!!!!!!!!");
      /***Measuremnt Jacobian matrix H and measurents vector ***/
      state_.last_update_time = time_compass_last_;
      Eigen::VectorXd meas_vec(6);
      Eigen::Matrix3d rot_compass_global = buf_rot_compass_global_.front();
      filter_global2local(pos_compass_global_, rot_compass_global, state_, meas_vec);//meas_vec =（状态变量-观测量）转换到局部坐标系的差值，姿态三个自由度 + pos三个自由度
      buf_rot_compass_global_.clear();
      mtx_compass.unlock(); // 释放读锁

      std::cout<<"======== state_ before new compass========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
      std::cout<<"========  meas_vec  ========"<<std::endl<< meas_vec <<std::endl;    
      // Eigen::Matrix< double, DIM_OF_STATES, 1 > d_state;//[9x1]
      d_state.setZero();      // 误差状态反馈到系统状态后,将误差状态清零
      Eigen::MatrixXd Hsub(6,9);
      Hsub.setZero();
      //6维观测（姿态的观测与姿态的状态变量都是3维李代数），姿态观测rho对状态变量rho的Jacobian是1，pos观测yxz对pos_end的Jacobian是1，观测对状态变量vel_end的Jacobian是0。
      Hsub << 1,0,0,0,0,0,0,0,0,
              0,1,0,0,0,0,0,0,0,
              0,0,1,0,0,0,0,0,0,
              0,0,0,1,0,0,0,0,0,
              0,0,0,0,1,0,0,0,0,
              0,0,0,0,0,1,0,0,0;          
      // construct measurement noise matrix
      Eigen::MatrixXd R_compass(6,6);
      R_compass<<COMPASS_OBSERVE_COV,0,0,0,0,0,
                 0,COMPASS_OBSERVE_COV,0,0,0,0,
                 0,0,COMPASS_OBSERVE_COV,0,0,0,
                 0,0,0,COMPASS_OBSERVE_COV,0,0,
                 0,0,0,0,COMPASS_OBSERVE_COV,0,
                 0,0,0,0,0,COMPASS_OBSERVE_COV; //////////////////////////////////////////////////////////////////注意噪声位置！！！

      /*** Error State Kalman Filter Update ***/
      // std::cout<<"========  state_.cov  ========"<<std::endl<< state_.cov <<std::endl;    
      auto temp         = Hsub * state_.cov * Hsub.transpose() + R_compass;//[6x6]=[6x9]*[9x9]*[9x6]+[6x6]
      // std::cout<<"========  temp  ========"<<std::endl<< temp <<std::endl;    
      Eigen::MatrixXd K = state_.cov * Hsub.transpose() * temp.inverse();//[9x6]=[9x9]*[9x6]*[6x6]
      // std::cout<<"========  K  ========"<<std::endl<< K <<std::endl;    
      //【Jacobian(Hsub)、g_state_.cov、K、solution都是so3相关变量】
      // ESKF(ErKF)：meas_vec是观测量差值，d_state_是状态变量差值，ESKF小册子给的是状态变量和观测量nominal值，但是残差值更合理，武大KF-GINS也是残差值
      // 如果每次观测量更新后都进行误差状态反馈，则Imu_Process不会影响d_state，它的值直到下一次观测量更新之前都是0，下式可以简化为：d_state = K * meas_vec
      d_state = K * meas_vec - K * Hsub * d_state;//[9x1] = [9x6]*[6x1]-[9x6]*[6x9]*[9x1]
      // std::cout<<"========d_state========"<<std::endl<< d_state <<std::endl;    
      // Eigen::Matrix3d rot_temp = SO3_Exp(d_state(0), d_state(1), d_state(2));
      // std::cout<<"========rot_temp========"<<std::endl<< rot_temp <<std::endl; 
      state_ = state_ + d_state;//【SO3+so3=SO3】

      /*** Covariance Update ***/
      I_KH = I_STATE - K * Hsub;//[9x9] = [9x9]-[9x6]*[6x9]
      state_.cov = I_KH * state_.cov * I_KH.transpose() + K * R_compass * K.transpose();//[9x9] = [9x9]*[9x9]*[9x9] + [9x6]*[6x6]*[6x9]
      // std::cout<<"======== state_.cov ========"<<std::endl<< state_.cov <<std::endl;    

      filter_flag_ += 2;
      std::cout<<"======== state_ after compass filter ========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
    }

    if(!buf_pos_tag_global_.empty() && mtx_tag.try_lock())//tag buffer没被占用 && 有新的tag观测
    {
      ROS_INFO("!!!!!!!!NEW Tag!!!!!!!!");
      /***Measuremnt Jacobian matrix H and measurents vector ***/
      state_.last_update_time = time_tag_last_;
      Eigen::VectorXd meas_vec(6);
      Eigen::Vector3d pos_tag_global = buf_pos_tag_global_.front();
      // Eigen::Matrix3d rot_tag_global = buf_rot_tag_global_.front();
      filter_global2local(pos_tag_global, rot_tag_global_, state_, meas_vec);//meas_vec =（观测量-状态变量predict）转换到局部坐标系的差值，姿态三个自由度 + pos三个自由度
      buf_pos_tag_global_.clear();
      mtx_tag.unlock(); // 释放读锁

      std::cout<<"======== state_ before new tag ========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
      std::cout<<"========  meas_vec  ========"<<std::endl<< meas_vec <<std::endl;    
      double latency = ros::Time::now().toSec() - time_tag_last_;//<s>
      if(latency > 0.1)//trigger the Out Of Sequenence Measurement method
      {
        ROS_INFO("!!!!!!!! OOSM !!!!!!!!");
      }
      // Eigen::Matrix< double, DIM_OF_STATES, 1 > d_state;//[9x1]
      d_state.setZero();      // 误差状态反馈到系统状态后,将误差状态清零
      Eigen::MatrixXd Hsub(6,9);
      Hsub.setZero();
      //6维观测（姿态的观测与姿态的状态变量都是3维李代数），姿态观测rho对状态变量rho的Jacobian是1，pos观测yxz对pos_end的Jacobian是1，观测对状态变量vel_end的Jacobian是0。
      Hsub  <<  1,0,0,0,0,0,0,0,0,
                0,1,0,0,0,0,0,0,0,
                0,0,1,0,0,0,0,0,0,
                0,0,0,1,0,0,0,0,0,
                0,0,0,0,1,0,0,0,0,
                0,0,0,0,0,1,0,0,0;     
      // construct measurement noise matrix
      Eigen::MatrixXd R_tag(6,6);
      // R_gnss = gnssdata.std.cwiseProduct(gnssdata.std).asDiagonal();
      R_tag<< TAG_OBSERVE_COV,0,0,0,0,0,
              0,TAG_OBSERVE_COV,0,0,0,0,
              0,0,TAG_OBSERVE_COV,0,0,0,
              0,0,0,TAG_OBSERVE_COV,0,0,
              0,0,0,0,TAG_OBSERVE_COV,0,
              0,0,0,0,0,TAG_OBSERVE_COV; 

      /*** Error State Kalman Filter Update ***/
      auto temp         = Hsub * state_.cov * Hsub.transpose() + R_tag;//[6x6]=[6x9]*[9x9]*[9x6]+[6x6]
      Eigen::MatrixXd K = state_.cov * Hsub.transpose() * temp.inverse();//[9x6]=[9x9]*[9x6]*[6x6]

      // std::cout<<"========  temp  ========"<<std::endl<< temp <<std::endl;    
      // std::cout<<"========  K  ========"<<std::endl<< K <<std::endl;    
      //【Jacobian(Hsub)、g_state_.cov、K、solution都是so3相关变量】
      // ESKF(ErKF)：meas_vec是观测量差值，d_state_是状态变量差值，ESKF小册子给的是状态变量和观测量nominal值，但是残差值更合理，武大KF-GINS也是残差值
      // 如果每次观测量更新后都进行误差状态反馈，则Imu_Process不会影响d_state，它的值直到下一次观测量更新之前都是0，下式可以简化为：d_state = K * meas_vec
      d_state = K * meas_vec - K * Hsub * d_state;//[9x1] = [9x6]*[6x1]-[9x6]*[6x9]*[9x1]
      // std::cout<<"========d_state========"<<std::endl<< d_state <<std::endl;    
      // Eigen::Matrix3d rot_temp = SO3_Exp(d_state(0), d_state(1), d_state(2));
      // std::cout<<"========rot_temp========"<<std::endl<< rot_temp <<std::endl; 
      // std::cout<<"========d_state========"<<std::endl<< d_state <<std::endl;    
      state_ = state_ + d_state;//【SO3+so3=SO3】

      /*** Covariance Update ***/
      I_KH = I_STATE - K * Hsub;//[9x9] = [9x9]-[9x6]*[6x9]
      state_.cov = I_KH * state_.cov * I_KH.transpose() + K * R_tag * K.transpose();//[9x9] = [9x9]*[9x9]*[9x9] + [9x6]*[6x6]*[6x9]
      // std::cout<<"======== state_.cov ========"<<std::endl<< state_.cov <<std::endl;    

      filter_flag_ += 4;
      std::cout<<"======== state_ after tag filter ========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
    }

    if(!buf_pos_uwb_global_.empty() && mtx_uwb.try_lock())//uwb buffer没被占用 && 有新的uwb观测
    {
      ROS_INFO("!!!!!!!!NEW UWB!!!!!!!!");
      state_.last_update_time = time_uwb_last_;
      /***Measuremnt Jacobian matrix H and measurents vector ***/
      Eigen::VectorXd meas_vec(6);
      Eigen::Vector3d pos_uwb_global = buf_pos_uwb_global_.front();
      filter_global2local(pos_uwb_global, rot_uwb_global_, state_, meas_vec);//meas_vec =（状态变量-观测量）转换到局部坐标系的差值，姿态三个自由度 + pos三个自由度 //rot_uwb_global_ = eye3
      buf_pos_uwb_global_.clear();
      mtx_uwb.unlock(); // 释放读锁

      std::cout<<"======== state_ before new UWB ========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
      std::cout<<"========  meas_vec  ========"<<std::endl<< meas_vec <<std::endl;    
      // Eigen::Matrix< double, DIM_OF_STATES, 1 > d_state;//[9x1]
      d_state.setZero();      // 误差状态反馈到系统状态后,将误差状态清零
      Eigen::MatrixXd Hsub(6,9);
      Hsub.setZero();
      //6维观测（姿态的观测与姿态的状态变量都是3维李代数），姿态观测rho对状态变量rho的Jacobian是1，pos观测yxz对pos_end的Jacobian是1，观测对状态变量vel_end的Jacobian是0。
      Hsub  <<  1,0,0,0,0,0,0,0,0,
                0,1,0,0,0,0,0,0,0,
                0,0,1,0,0,0,0,0,0,
                0,0,0,1,0,0,0,0,0,
                0,0,0,0,1,0,0,0,0,
                0,0,0,0,0,1,0,0,0;     
      // construct measurement noise matrix
      Eigen::MatrixXd R_uwb(6,6);
      // R_gnss = gnssdata.std.cwiseProduct(gnssdata.std).asDiagonal();
      R_uwb <<UWB_OBSERVE_COV,0,0,0,0,0,
              0,UWB_OBSERVE_COV,0,0,0,0,
              0,0,UWB_OBSERVE_COV,0,0,0,
              0,0,0,UWB_OBSERVE_COV,0,0,
              0,0,0,0,UWB_OBSERVE_COV,0,
              0,0,0,0,0,UWB_OBSERVE_COV; 

      /*** Error State Kalman Filter Update ***/
      auto temp         = Hsub * state_.cov * Hsub.transpose() + R_uwb;//[6x6]=[6x9]*[9x9]*[9x6]+[6x6]
      Eigen::MatrixXd K = state_.cov * Hsub.transpose() * temp.inverse();//[9x6]=[9x9]*[9x6]*[6x6]
      // std::cout<<"========  temp  ========"<<std::endl<< temp <<std::endl;    
      // std::cout<<"========  K  ========"<<std::endl<< K <<std::endl;    
      //【Jacobian(Hsub)、g_state_.cov、K、solution都是so3相关变量】
      // ESKF(ErKF)：meas_vec是观测量差值，d_state_是状态变量差值，ESKF小册子给的是状态变量和观测量nominal值，但是残差值更合理，武大KF-GINS也是残差值
      // 如果每次观测量更新后都进行误差状态反馈，则Imu_Process不会影响d_state，它的值直到下一次观测量更新之前都是0，下式可以简化为：d_state = K * meas_vec
      d_state = K * meas_vec - K * Hsub * d_state;//[9x1] = [9x6]*[6x1]-[9x6]*[6x9]*[9x1]
      // std::cout<<"========d_state========"<<std::endl<< d_state <<std::endl;    
      // Eigen::Matrix3d rot_temp = SO3_Exp(d_state(0), d_state(1), d_state(2));
      // std::cout<<"========rot_temp========"<<std::endl<< rot_temp <<std::endl; 
      state_ = state_ + d_state;//【SO3+so3=SO3】

      /*** Covariance Update ***/
      I_KH = I_STATE - K * Hsub;//[9x9] = [9x9]-[9x6]*[6x9]
      state_.cov = I_KH * state_.cov * I_KH.transpose() + K * R_uwb * K.transpose();//[9x9] = [9x9]*[9x9]*[9x9] + [9x6]*[6x6]*[6x9]
      // std::cout<<"========  state_.cov  ========"<<std::endl<< state_.cov <<std::endl;    

      filter_flag_ += 8;
      std::cout<<"======== state_ after new UWB ========"<<std::endl<< state_.rot_end <<std::endl<< state_.pos_end <<std::endl<< state_.vel_end <<std::endl;    
    }


    filter_local2global(state_, pos_filter_global_, rot_filter_global_);

    double sx = sqrt(state_.rot_end(0,0)*state_.rot_end(0,0) + state_.rot_end(1,0)*state_.rot_end(1,0));//rot_filter_global_.at<double>(0,0)
    double roll_filter = atan2(state_.rot_end(2, 1), state_.rot_end(2, 2)) * R2D;
    double pitch_filter = atan2(-state_.rot_end(2, 0), sx) * R2D;
    double yaw_filter = atan2(state_.rot_end(1, 0), state_.rot_end(0, 0)) * R2D; 
    printf("filter local possss =========================== %lf %lf %lf \n", state_.pos_end(0), state_.pos_end(1), state_.pos_end(2));
    printf("filter eular angles =========================== %lf %lf %lf \n", roll_filter, pitch_filter, yaw_filter);

    //获得当前姿态欧拉角形式，用于记录log
    // double sy = sqrt(rot_filter_global_(0,0)*rot_filter_global_(0,0) + rot_filter_global_(1,0)*rot_filter_global_(1,0));//rot_filter_global_.at<double>(0,0)
    // bool singular = sy < 1e-6;
    // if(!singular)
    // {
    //   roll_filter_ = atan2(rot_filter_global_(2, 1), rot_filter_global_(2, 2)) * R2D;
    //   pitch_filter_ = atan2(-rot_filter_global_(2, 0), sy) * R2D;
    //   yaw_filter_ = atan2(rot_filter_global_(1, 0), rot_filter_global_(0, 0)) * R2D; 
    // }
    // else
    // {    
    //   roll_filter_ = atan2(-rot_filter_global_(1, 2), rot_filter_global_(1, 1)) * R2D;    
    //   pitch_filter_ = atan2(-rot_filter_global_(2, 0), sy) * R2D; 
    //   yaw_filter_ = 0;
    // }
    // filter_count_++;
    // dump_filter_state_to_log(m_state_fp);
    // filter_flag_ = 0;

#ifdef OUTPUT_FOR_PAPER
  std::string bufferfile = std::to_string(state_.last_update_time) + "," + std::to_string(state_.pos_end(0))+ "," 
                         + std::to_string(state_.pos_end(1))+ "," + std::to_string(state_.pos_end(2)) + "\n";
  const char* buffer_file = bufferfile.c_str();
  std::cout<<"fusion buffer_file: "<< buffer_file <<std::endl;
  fwrite(buffer_file, 1, 50, fp_fusion_pos);
#else
    /******* Publish Odometry-filter-local ******/
    Eigen::Quaterniond qe = Eigen::Quaterniond(state_.rot_end);
    qe.normalize();
    geometry_msgs::Quaternion qg;
    tf::quaternionEigenToMsg(qe, qg);//#include <eigen_conversions/eigen_msg.h>不能少，CMakeList添加eigen_conversions
    odom_filter_local.header.frame_id = "world";
    odom_filter_local.header.stamp = ros::Time::now(); // ros::Time().fromSec(state_.last_update_time);
    odom_filter_local.pose.pose.orientation.x = qg.x;
    odom_filter_local.pose.pose.orientation.y = qg.y;
    odom_filter_local.pose.pose.orientation.z = qg.z;
    odom_filter_local.pose.pose.orientation.w = qg.w;
    odom_filter_local.pose.pose.position.x = state_.pos_end(0);
    odom_filter_local.pose.pose.position.y = state_.pos_end(1);
    odom_filter_local.pose.pose.position.z = state_.pos_end(2);
    pub_odom_filter_local.publish( odom_filter_local );

    /******* Publish Odometry-filter-global ******/
    qe = Eigen::Quaterniond(rot_filter_global_);
    qe.normalize();
    tf::quaternionEigenToMsg(qe, qg);
    odom_filter_global.header.frame_id = "world";
    odom_filter_global.header.stamp = ros::Time::now(); 
    odom_filter_global.pose.pose.orientation.x = qg.x;
    odom_filter_global.pose.pose.orientation.y = qg.y;
    odom_filter_global.pose.pose.orientation.z = qg.z;
    odom_filter_global.pose.pose.orientation.w = qg.w;
    odom_filter_global.pose.pose.position.x = pos_filter_global_(0);
    odom_filter_global.pose.pose.position.y = pos_filter_global_(1);
    odom_filter_global.pose.pose.position.z = pos_filter_global_(2);
    pub_odom_filter_global.publish( odom_filter_global );
#endif

    double time_end = ros::Time::now().toSec();
    cost_time_ = time_end - time_start;
    std::cout<<"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFilter processing time cost = "<< cost_time_ <<std::endl;    
    rate_filter.sleep();
  }
}

void initialization_process()
{
  ros::Rate rate_init(1);//初始化阶段数据收集频率

  //【为了平均不同传感器姿态计算anchor，被迫引入compass_last_enu_、yaw_reg_enu_、yaw_tag_enu_，filter融合时不用一维yaw，使用三维rot】
  ROS_INFO("////////////////////////Anchor calculation start//////////////////////////");
  /* 
  初始化逻辑：认为gnss和compass近似是一直都有的，再加上其他任意一种可提供位置的数据，就可以启动初始化进程。
  初始化过程中给位置和姿态分别计数。
  */
  while(!uwbbase_ready_ && !tagbase_ready_)
  {
    ROS_INFO("data not ready, waiting......");
    rate_init.sleep();
    // ros::spinOnce();
  }
  double time_start = ros::Time::now().toSec();
  int count_pos = 0;
  int count_yaw = 0;
  ROS_INFO("initial anchor_transition in progress...");
  while(ros::ok())//【计算anchor】
  {
    //以1hz频率加入gnss、compass、tag、uwb新数据，求平均作为 anchor_transition
    if(!buf_pos_gnss_global_.empty() && star_num_gnss_ > threshold_gnss_star_ && mtx_gnss.try_lock())//gnss buffer没被占用 && 有新的gnss观测 && 搜星数>阈值
    {
      Eigen::Vector3d pos_gnss_global = buf_pos_gnss_global_.front();
      buf_pos_gnss_global_.clear();
      mtx_gnss.unlock(); // 释放读锁
      anchor_transition_xyz_ += pos_gnss_global; //配准和gnss位置都是基于ECEF
      count_pos++; 
      std::cout<< "collecting gnss data, count_pos = " << count_pos << ", count_yaw = "<< count_yaw <<std::endl;  
      std::cout<< "pos_gnss_global = " <<std::endl<< pos_gnss_global <<std::endl;  
    }   

    if(!buf_compass_yaw_enu_.empty() && mtx_compass.try_lock())//compass buffer没被占用 && 有新的compass观测
    {
      double compass_enu = buf_compass_yaw_enu_.front();
      buf_compass_yaw_enu_.clear();
      mtx_compass.unlock(); // 释放读锁
      yaw_transition_enu_ += compass_enu*R2D;//compass 基准是？？
      count_yaw++; 
      std::cout<< "collecting compass data, count_pos = " << count_pos << ", count_yaw = "<< count_yaw <<std::endl;  
      std::cout<< "compass_enu = " << (compass_enu*R2D) <<std::endl;  
    }

    if(!buf_pos_tag_global_.empty() && mtx_tag.try_lock())//tag buffer没被占用 && 有新的tag观测
    {
      Eigen::Vector3d pos_tag_global = buf_pos_tag_global_.front();
      // double yaw_tag_enu = buf_yaw_tag_enu_.front();
      buf_pos_tag_global_.clear();
      // buf_rot_tag_global_.clear();
      mtx_tag.unlock(); // 释放读锁
      anchor_transition_xyz_ += pos_tag_global;
      // yaw_transition_enu_ += yaw_tag_enu;
      count_pos++; 
      // count_yaw++; 
      std::cout<< "collecting tag data, count_pos = " << count_pos << ", count_yaw = "<< count_yaw <<std::endl;  
      std::cout<< "pos_tag_global = " <<std::endl<< pos_tag_global <<std::endl; 
      // std::cout<< "yaw_tag_enu = " << yaw_tag_enu <<std::endl;       
    }

    if(!buf_pos_uwb_global_.empty() && mtx_uwb.try_lock())//uwb buffer没被占用 && 有新的uwb观测
    {
      Eigen::Vector3d pos_uwb_global = buf_pos_uwb_global_.front();
      buf_pos_uwb_global_.clear();
      mtx_uwb.unlock(); // 释放读锁
      anchor_transition_xyz_ += pos_uwb_global;
      count_pos++; 
      std::cout<< "collecting tag data, count_pos = " << count_pos << ", count_yaw = "<< count_yaw <<std::endl;  
      std::cout<< "pos_uwb_global = " <<std::endl<< pos_uwb_global <<std::endl; 
    }
    std::cout<< "count_pos = " << count_pos << ", count_yaw = "<< count_yaw <<std::endl;  
    if(count_pos > threshold_anchor_pos_ && count_yaw > threshold_anchor_yaw_)
      break;
    rate_init.sleep();
    // ros::spinOnce();
  }
  yaw_transition_enu_ /= count_yaw;
  anchor_transition_xyz_ /= count_pos;
  anchor_transition_blh_ = ecef_xyz2blh(anchor_transition_xyz_);

  double lat = anchor_transition_blh_.x()*D2R, lon = anchor_transition_blh_.y()*D2R;//经纬高使用 anchor_transition_blh_
  double sin_lat = sin(lat), cos_lat = cos(lat);
  double sin_lon = sin(lon), cos_lon = cos(lon);
  Eigen::Matrix3d R_ecef_enu;
  R_ecef_enu <<  -sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon,
                  cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon,
                  0      ,  cos_lat        , sin_lat;
  double sin_yaw_diff = std::sin(yaw_transition_enu_*D2R);//从local到enu：sin(yaw);从enu到local：sin(-yaw)!!!!!!!!!!!!!!!!!!!!!!!!!!
  double cos_yaw_diff = std::cos(yaw_transition_enu_*D2R);
  Eigen::Matrix3d R_enu_local;
  R_enu_local << cos_yaw_diff, -sin_yaw_diff, 0,
                 sin_yaw_diff,  cos_yaw_diff, 0,
                 0           ,  0           , 1;
  R_ecef_local_transition_ = R_ecef_enu * R_enu_local;
  transition_ready_ = true;
  double time_end = ros::Time::now().toSec();
  ROS_INFO("anchor_transition finish! cost time = %d ms", time_end-time_start);//为什么显示毫秒？？？？？？
  std::cout<< "yaw_transition_enu_ = "               << yaw_transition_enu_    <<std::endl;    
  std::cout<< "anchor_transition_xyz_ = " <<std::endl<< anchor_transition_xyz_ <<std::endl;
  std::cout<< "anchor_transition_blh_ = " <<std::endl<< anchor_transition_blh_ <<std::endl;
  std::cout<<"!!!!!!!!!!!!  R_enu_local  !!!!!!!!!!!!!"<<std::endl<< R_enu_local <<std::endl;    

  std::thread filter_process{filtering_process};
  filter_process.detach();//异步、非阻塞
}


int main(int argc, char **argv) 
{
  ros::init(argc, argv, "vtol_msn");
  ros::NodeHandle nh("~");
  ROS_INFO("////////////////////////Initialization//////////////////////////");

  std::string tag_file;//不含后缀名
  nh.getParam("tag_file", tag_file);
  std::cout<< "tag_file:"<< tag_file <<std::endl;
  std::string uwb_file;//不含后缀名
  nh.getParam("uwb_file", uwb_file);
  std::cout<< "uwb_file:"<< uwb_file <<std::endl;
  nh.getParam("gnss_cov", GNSS_OBSERVE_COV);
  std::cout<< "GNSS_OBSERVE_COV:"<< GNSS_OBSERVE_COV <<std::endl;

  nh.getParam("tag_cov", TAG_OBSERVE_COV);
  std::cout<< "TAG_OBSERVE_COV:"<< TAG_OBSERVE_COV <<std::endl;
  nh.getParam("uwb_cov", UWB_OBSERVE_COV);
  std::cout<< "UWB_OBSERVE_COV:"<< UWB_OBSERVE_COV <<std::endl;
  nh.getParam("R_init_COV", R_init_COV);
  std::cout<< "R_init_COV:"<< R_init_COV <<std::endl;
  nh.getParam("P_init_COV", P_init_COV);
  std::cout<< "P_init_COV:"<< P_init_COV <<std::endl;
  nh.getParam("V_init_COV", V_init_COV);
  std::cout<< "V_init_COV:"<< V_init_COV <<std::endl;
  nh.getParam("omega_noise", COV_OMEGA_NOISE_DIAG);
  std::cout<< "COV_OMEGA_NOISE_DIAG:"<< COV_OMEGA_NOISE_DIAG <<std::endl;
  nh.getParam("vel_noise", COV_VEL_NOISE_DIAG);
  std::cout<< "COV_VEL_NOISE_DIAG:"<< COV_VEL_NOISE_DIAG <<std::endl;  
  nh.getParam("acc_noise", COV_ACC_NOISE_DIAG);
  std::cout<< "COV_ACC_NOISE_DIAG:"<< COV_ACC_NOISE_DIAG <<std::endl;


  /*
  //sensor_msgs/NavSatFix，又包含sensor_msgs/NavSatStatus = Navigation Satellite fix status + Global Navigation Satellite System service type
  // status: 
  // int8 STATUS_NO_FIX=-1
  // int8 STATUS_FIX=0        # 通过QGC的EKF2_REQ_NSATS设置阈值(默认6)
  // int8 STATUS_SBAS_FIX=1   # with satellite-based augmentation
  // int8 STATUS_GBAS_FIX=2   # with ground-based augmentation
  // service: uint16 SERVICE_GPS=1, uint16 SERVICE_GLONASS=2, uint16 SERVICE_COMPASS=4, uint16 SERVICE_GALILEO=8
  */
  //sensor_msgs/NavSatFix, 基于WGS84的经纬[deg]高[m]，GVINS用的是ECEF坐标WGS84标准
  gnss_gt_sub = nh.subscribe("/ublox_driver/receiver_lla", 10, gnss_gt_callback);//sensor_msgs/NavSatFixg格式 【10hz】【差分ublox作为真值】

  //当前三种数据全部来自于FDI，也可以用ExactTime，但是考虑到使用其他数据源的可能性选择了ApproximateTime
  // message_filters::Subscriber<sensor_msgs::NavSatFix> gnss_sub(nh, "/raw_gnss", 10);//sensor_msgs/NavSatFixg格式【1hz】
  // message_filters::Subscriber<fdilink_ahrs::satellite> gnss_sat_sub(nh, "/satellite", 10);//【1hz】
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::NavSatFix, fdilink_ahrs::satellite> MySyncPolicy;
  // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), gnss_sub, gnss_sat_sub);
  // sync.registerCallback(boost::bind(&gnss_callback, _1, _2));

  imu_sub = nh.subscribe("/imu", 200, imu_callback, ros::TransportHints().tcpNoDelay());//【100hz】
  compass_sub = nh.subscribe("/mag_pose", 10, compass_callback);//【5hz】 
  // tag_sub = nh.subscribe("/tag_detections", 10, tag_callback); //【5hz】
  // uwb_sub = nh.subscribe("/nlink_linktrack_nodeframe2", 10, uwb_callback); //【5hz】      
  uwb_sub = nh.subscribe("/nlink_linktrack_nodeframe2_0", 10, uwb_callback); //【5hz】      

#ifdef OUTPUT_FOR_PAPER
  std::string filename_gnss_gt = "/home/zbh/gt_data.dat"; 
  const char* file_name_gnss_gt = filename_gnss_gt.c_str();
  fp_gnss_gt = fopen(file_name_gnss_gt,"w");
  std::string filename_sensor_pos = "/home/zbh/sensor_data.dat"; 
  const char* file_name_sensor_pos = filename_sensor_pos.c_str();
  fp_sensor_pos = fopen(file_name_sensor_pos,"w");
  std::string filename_fusion_pos = "/home/zbh/fusion_data.dat"; 
  const char* file_name_fusion_pos = filename_fusion_pos.c_str();
  fp_fusion_pos = fopen(file_name_fusion_pos,"w");
#else
  pub_odom_filter_local = nh.advertise<nav_msgs::Odometry>("/vtol_msn/odom_filter_local", 1000);//【不再处理姿态数据！只发布xyz！】
  pub_odom_filter_global = nh.advertise<nav_msgs::Odometry>("/vtol_msn/odom_filter_global", 1000);
#endif

  initializel_uwb(uwb_file);//先执行，以获得R_ecef_enu_
  initializel_tag(tag_file);

  std::thread initial_process{initialization_process};
  initial_process.detach();//异步、no阻塞

  ros::spin();
}
