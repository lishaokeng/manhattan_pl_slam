# manhattan_pl_slam

使用曼哈顿假设全局约束的点线vslam

## 介绍

- 本项目为点线视觉slam项目。

### 项目发展历程

1. 由vins-mono去除imu改进得到单目vslam项目原型；
2. 手动实现相机模型类（畸变去畸变、坐标转换等）；
3. 添加轮速计传感器和后端尺度约束，解决单目尺度漂移问题（尺度约束由本人推导并实现的曲线模长尺度，网上无资料）；
4. 单目初始化改为使用轮速计位姿进行初始化（提高初始化成功率）；
5. 弃用opencv图像金字塔算法，手动实现并优化图像金字塔算法，耗时为opencv的2/3且比opencv耗时稳定（参考BASALT-SLAM）；
6. 弃用opencv光流跟踪算法，手动实现逆向光流，耗时为opencv的1/4(参考BASALT-SLAM)；
7. 弃用vins的前端BA优化，改为手动实现逆向BA，耗时为ceres的1/2（参考半直接法SVO）；
8. 添加ELSED线段检测和ZNCC线段跟踪、曼哈顿结构线相关重投影约束（参考ELSED、manhattan-slam、structure-slam）；
9. 添加特征点深度滤波器，解决KLT光流跟踪召回率不够问题（参考半直接法SVO和直接法DSO）；
10. 添加普吕克线段初始化和相关重投影约束（参考PL-VIO）；
11. 由普吕克线段改进为曼哈顿假设，解决原版structure-slam线段初始化准确率低导致优化偏离问题（无参考）；
12. 改进ELSED线段提取策略，对已初始化线段使用edge-drawing锚点进行投影和线段提取跟踪，极大提高ELSED线段检测召回率（无参考）；
13. 弃用后端边缘化约束（耗时过高且对定位帮助不大）；
14. 由俯视图语义分割结果建图，重投影到曼哈顿坐标系去除avm平面假设，解决平面假设带来的不同帧点云漂移问题（参考激光slam cartographer子图拼接和更新）；

## 目录如下

- camera_model: 相机模型
- depth_filter: 深度滤波器
- elsed: elsed线段检测
- inverse_opticalflow: 特征点逆向光流
- slam_video: 点线slam展示视频
