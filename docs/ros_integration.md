# ROS 集成技能

## 编码规范

- 使用 C++17 标准。
- ROS2 或 ROS1 编码规范，消息类型、话题名称、坐标系遵循 REP-105、REP-103。
- TF 坐标必须定义 parent-child 关系，禁止未发布 TF 直接使用坐标转换。
- 节点、话题、服务、动作名称统一小写下划线，避免硬编码。

## 图像与点云处理

- OpenCV 图像、点云数据和 ROS 消息转换（`cv_bridge`、`sensor_msgs/Image`、`sensor_msgs/PointCloud2`）必须做异常捕获。
- 判断图像空指针，防止图像解码崩溃。
- 区分仿真图像数据和实机相机数据，两套接口兼容。
- 点云处理使用 PCL，转换时注意字段名和数据类型，避免内存泄漏。

## 参数管理

- 文件路径、相机内参、标定参数、话题名、节点名全部写入 YAML 配置文件。
- 严格禁止代码内硬编码参数。
- C++ 通过参数服务器加载配置。
- YAML 参数使用命名空间分组，加载后打印关键参数确认生效。

## 编译与依赖

- 区分 CMakeLists 链接依赖（OpenCV、PCL、image-transport、tf2、nav2、behavior_tree_cpp）、package.xml 依赖项、colcon-build 编译报错。
- 自动区分环境缺失、版本不兼容、语法错误。
- 自动补全 package.xml 和 CMakeLists 缺失依赖声明。

## 仿真与实机分离

- 仿真节点和实机节点代码分离。
- 可通过 launch 文件切换模式。
- 自动识别 RViz 可视化报错、TF 树断裂、话题无数据、帧率过低问题。
- 仿真中使用 Gazebo / Isaac Sim，实机使用真实传感器驱动。

## 代码结构

- 视觉算法模块（图像滤波、目标检测、轮廓识别、相机标定、手眼标定）单独封装 C++ 类。
- 代码重构时，将视觉算法、ROS 消息收发、硬件 IO 三层代码解耦。
- 禁止业务代码和 ROS 回调函数混杂。
- 使用插件机制（`pluginlib`）动态加载导航、定位、行为树节点。

## 日志与调试

- 添加日志输出（RCLCPP_INFO / ROS_INFO），标记图像接收时间戳。
- 视觉处理耗时做计时输出，优化卡顿代码。
- 导航与 SLAM 节点输出关键指标（定位协方差、地图更新频率、规划时间）。

## 代码输出规则

- 不一次性输出完整大型项目代码，拆分类文件、launch 文件、配置文件。
- 注释区分相机标定参数、图像预处理参数、算法阈值。
- 只输出必要的硬编码，非必须尽量避免。

## SLAM 集成规范

### 传感器与数据流

- 激光雷达：`sensor_msgs/LaserScan`，频率不低于 10Hz，距离范围匹配实机。
- 深度相机：`sensor_msgs/Image` + `sensor_msgs/CameraInfo`，用于视觉 SLAM 或 RGB-D。
- IMU：`sensor_msgs/Imu`，提供角速度、线加速度，频率不低于 200Hz。
- 里程计：`nav_msgs/Odometry`，提供轮式/足式里程计。
- 所有传感器数据必须带时间戳同步（`message_filters` 或 `approx_time_synchronizer`）。

### TF 要求

- 地图坐标系：`map`
- 里程计坐标系：`odom`
- 机器人基座：`base_link`（或 `base_footprint`）
- 激光雷达：`laser`
- 相机：`camera_link`
- `map -> odom` 由 SLAM 或定位节点发布，`odom -> base_link` 由里程计发布。
- TF 发布频率不低于 50Hz，避免树断裂。

### 常用 SLAM 方案

- 2D 激光：Cartographer、Gmapping、slam_toolbox
- 视觉/惯性：VINS-Fusion、ORB-SLAM3、RTAB-Map
- 推荐使用 `slam_toolbox` 作为默认 2D SLAM，配置文件中设置地图分辨率（0.05m）、更新频率（5Hz）。
- 保存/加载地图使用 `map_server`，地图格式 `nav_msgs/OccupancyGrid`。

### 定位

- 已知地图下使用 AMCL（自适应蒙特卡洛定位），配置粒子数、重采样策略。
- 视觉/惯性定位需输出 `geometry_msgs/PoseWithCovarianceStamped`。
- 定位失败时（协方差超限、长时间无匹配），触发安全行为：减速或停车。

### 导航集成规范

#### 框架选择

- ROS1：move_base（global planner + local planner + recovery behaviors）。
- ROS2：Nav2（planner server + controller server + behavior tree nav）。
- 导航输出统一为 `geometry_msgs/Twist`（cmd_vel），频率 20Hz。

#### 与全身运动控制器接口

- 导航层输出 `cmd_vel`（vx, vy, omega）给高层指挥节点。
- 全身运动控制器接收期望速度并转换为关节力矩。
- 接口节点职责：
  1. 订阅 `cmd_vel`，做平滑和限幅。
  2. 将速度映射到机器人坐标系。
  3. 发布给 RL 策略作为观测或直接作为目标。
  4. 检测到异常（如滑移、翻倒）时上报导航层触发恢复行为。

#### 代价地图

- 全局代价地图：静态地图 + 障碍物层，分辨率与地图一致。
- 局部代价地图：滚动窗口（通常 3m x 3m），包含实时障碍物。
- 层配置：static_layer、obstacle_layer、inflation_layer。
- 参数写入 YAML，禁止硬编码。

#### 路径规划

- 全局规划器：NavFn、A*、Theta*。
- 局部规划器：DWA、TEB、MPC。
- 设置最大速度、加速度、角速度，确保不超过机器人物理极限。
- 规划频率：全局 1Hz，局部 10-20Hz。

#### 恢复行为

- 当局部规划失败或定位丢失，触发恢复行为：
  1. 清除代价地图
  2. 原地旋转 360°
  3. 后退
  4. 急停并请求人工干预
- 恢复行为通过行为树节点管理，优先级最高。

## 行为树集成规范

### 框架

- ROS2 使用 `BehaviorTree.CPP`（v4.x）+ `Groot2` 作为编辑和监控工具。
- 行为树文件为 XML 格式，放在 `config/behavior_trees/` 目录。

### 节点类型

- **Action 节点**：执行具体任务（如导航到点、抓取、发布 cmd_vel、切换控制器模式）。
- **Condition 节点**：检查状态（电池电量、定位健康、任务标志）。
- **Decorator 节点**：重试、超时、强制成功/失败。
- **Control 节点**：Sequence、Fallback、Parallel。

### 与导航和运动控制对接

- Action 节点调用 Nav2 的 `NavigateToPose` 或 `FollowPath` 动作客户端。
- 运动控制节点订阅行为树发出的指令，如 `go_to_pose`、`rotate_in_place`、`dock`。
- 全身运动控制器作为底层执行器，接收来自行为树的离散命令和导航层的连续速度。
- 行为树节点内禁止直接操作硬件或仿真物理接口，必须通过 ROS 服务/动作。

### 典型任务流程

1. 任务启动：Condition 检查电量 > 30%。
2. 导航到 A 点：Action 节点调用 Nav2。
3. 执行操作：Action 节点触发机械臂或相机。
4. 返回充电桩：Action 节点导航 dock。
5. 紧急停止：Decorator 监控 `estop` 话题，一旦触发立即中断当前分支并执行安全态。

### 代码实现

- 每个 Action 节点实现 `tick()` 方法，返回 `SUCCESS`/`FAILURE`/`RUNNING`。
- 节点内部使用 ROS 异步通信，禁止阻塞 `tick()` 超过控制周期（>100ms 必须使用异步状态机）。
- 使用 `BT::Blackboard` 交换数据，如目标位姿、当前速度、任务 ID。
- 行为树加载通过 `BT::BehaviorTreeFactory`，参数由 ROS 参数服务器提供。

### 调试与监控

- 使用 Groot2 实时监控树执行状态。
- 记录节点执行时间和失败原因到日志。
- 每个关键节点发布状态到 `bt_status` 话题，便于 RViz 或诊断工具显示。

## 导航、SLAM、行为树与仿真的协同

- 仿真环境中，SLAM 节点可接入 Gazebo/Isaac Sim 的 ground truth 里程计和虚拟激光雷达。
- 行为树在仿真和实机中使用同一 XML 文件，仅通过参数区分话题名。
- 集成测试时验证：从起点导航到目标，SLAM 建图，行为树分支切换，全身运动控制器速度跟踪。
- 性能要求：导航更新周期 50ms，SLAM 更新周期 200ms，行为树 tick 周期 100ms。
