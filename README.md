# VTOL_MSN_TakeOff_And_Landing

VTOL aircraft Multi-Sensor Navigation, in Take Off And Landing scenario.
Based on the Error-State Kalman Filter, a fusion navigation framework that combines INS, GNSS, Vision, and UWB modules.  
 <img src="(https://github.com/Space-Exploration-UAVTeam/VTOL_MSN_TakeOff_And_Landing/blob/master/imgs/123.png)  width="1200" />  

## 1. Prerequisites
### 1.1 ROS1
This package is tested under ROS Kinetic/Noetic.  

### 1.2 u-blox ZED-F9P
RTK for ground truth.  
https://github.com/HKUST-Aerial-Robotics/ublox_driver

### 1.3 FDI G90 module
Three-in-one sensor module that combines single-point positioning GNSS, IMU, and compass.  
https://github.com/SHUNLU-1/fdilink_ahrs 【Not publicly disclosed by the official, included with the purchase of the product.】

### 1.4 hikrobot MVS driver & hikrobot_ros1
https://www.hikrobotics.com/en/machinevision/service/download?module=0  
https://github.com/Space-Exploration-UAVTeam/hikrobot_ros1  

### 1.5 Nooploop LinkTrack
https://github.com/nooploop-dev/nlink_parser

## 2. Build 
Clone the repository to your catkin workspace (for example `~/catkin_ws/`):
```
cd ~/catkin_ws/src/
git clone https://github.com/Space-Exploration-UAVTeam/VTOL_MSN_TakeOff_And_Landing.git
```
Then build the package with:
```
cd ~/catkin_ws/
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Run
```
roslaunch vtol_msn vtol_msn.launch
```
