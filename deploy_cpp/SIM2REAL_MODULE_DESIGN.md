# Parkour 部署：Sim2Real 模块设计

日期：2026-06-05  
范围：`deploy_cpp` 当前 sim2sim / deploy 框架基础上的 sim2real 补齐方案。

---

## 1. 当前结论

当前 `deploy_cpp` 已经具备较完整的 **sim2sim + policy deploy 主框架**：

- 三段 JIT policy 推理：`heightmap_encoder`、`history_encoder`、`actor_backbone`。
- C++ `deploy_node` 主控。
- MuJoCo topic bridge。
- 真机电机 driver 外壳。
- 状态机：`IDLE`、`STAND_UP`、`RL`、`JOINT_DAMPING`、`RETURN_DEFAULT`、`JOINT_SWEEP`、`SINGLE_STEP_RL`。
- 键盘、手柄、UDP 输入。
- `/height_measurements`、`/fast_livo2/state6_imu_prop`、`/parkour/goal_yaw` 消费端。
- `/joint_states` 可视化。

但当前 sim2real 还缺少几个关键“真机生产端 / 桥接端”模块：

1. 真实 LiDAR / 点云 / 地图到 132 维 heightmap 的节点。
2. 真实 IMU / Odom / TF 到 policy IMU 输入的桥接节点。
3. 目标点 / waypoint / 导航接口。
4. `/cmd_vel` 或标准导航速度命令接入。
5. 真机 safety supervisor。
6. 真机日志、回放、健康监控。

尤其是“目标点接口”目前还没有完整做。当前只有 `/parkour/goal_yaw`，它只是给 policy 的 yaw 观测，不是完整的目标点或导航接口。

---

## 2. 当前已有模块

### 2.1 Policy Runner

相关文件：

- `deploy_cpp/include/parkour_policy_runner.h`
- `deploy_cpp/src/parkour_policy_runner.cpp`
- `deploy_cpp/include/robot_config.h`

当前 policy 输入大致为：

```text
commands:       vx, vy, yaw_rate
imu:            angular velocity + projected gravity
joint state:    q, dq
heightmap:      132 floats
goal_yaw:       [0, delta_yaw, delta_next_yaw]
```

policy runner 内部已经完成：

- heightmap encoder 推理。
- proprio history encoder 推理。
- actor backbone 推理。
- action scale / action clip。
- action 转目标关节位置。

sim2real 的重点不是重写 policy，而是将真实传感器、定位、地图和导航目标适配成这些输入。

---

### 2.2 Motor Driver

相关文件：

- `deploy_cpp/include/motor_driver.h`
- `deploy_cpp/src/motor_driver.cpp`

当前真机 motor driver 已经具备：

- 双串口。
- Unitree GO-M8010-6 SDK。
- motor id / port / reverse / gear ratio 映射。
- encoder offset calibration。
- joint-side 与 motor-side 转换。
- position PD、damping、zero torque。
- joint position / velocity / torque / temperature / error cache。

后续需要重点补：

- 电机反馈超时保护。
- 温度 / error code / torque watchdog。
- command rate 检查。
- motor health diagnostic topic。

---

### 2.3 State Machine 与遥控输入

主文件：

- `deploy_cpp/src/deploy_node.cpp`

当前已有状态：

```text
IDLE
STAND_UP
RL
JOINT_DAMPING
RETURN_DEFAULT
JOINT_SWEEP
SINGLE_STEP_RL
```

当前 command 来源：

1. Keyboard
2. UDP
3. Joy

当前还没有标准 `/cmd_vel`，因此自动导航或目标点控制还不能直接接入。

---

### 2.4 Sim Producer

MuJoCo 仿真节点：

- `deploy_cpp/sim/mujoco_sim_node.py`

当前 sim 会发布：

```text
/mujoco/joint_state
/fast_livo2/state6_imu_prop
/height_measurements
/parkour/goal_yaw
/joint_states
```

真机 sim2real 时，需要用真实节点替代 MuJoCo 中的这些 producer。

---

## 3. 当前 ROS Topic 状态

### 3.1 已有 topic

| Topic | 方向 | 类型 | 说明 |
|---|---|---|---|
| `/mujoco/joint_cmd` | deploy -> sim | `std_msgs/Float32MultiArray` | MuJoCo 电机目标 |
| `/mujoco/joint_state` | sim -> deploy | `std_msgs/Float32MultiArray` | MuJoCo 关节反馈 |
| `/fast_livo2/state6_imu_prop` | sim/real -> deploy | `std_msgs/Float32MultiArray` | `[wx, wy, wz, gx, gy, gz]` |
| `/height_measurements` | sim/real -> deploy | `std_msgs/Float32MultiArray` | 132 维 heightmap |
| `/parkour/goal_yaw` | sim/real -> deploy | `std_msgs/Float32MultiArray` | `[0, delta_yaw, delta_next_yaw]` |
| `/joint_states` | deploy/sim -> RViz | `sensor_msgs/JointState` | 机器人关节可视化 |
| `/joy` | joy_node -> deploy | `sensor_msgs/Joy` | 手柄输入 |

### 3.2 当前不存在的接口

目前没有发现以下接口：

```text
/goal_pose
/waypoints
/cmd_vel
Nav2 NavigateToPose action
Nav2 NavigateThroughPoses action
geometry_msgs/PoseStamped
geometry_msgs/Twist
nav_msgs/Path
自定义 .msg/.srv/.action
```

因此严格来说：

- “目标 yaw 观测接口”存在：`/parkour/goal_yaw`。
- “目标点接口”不存在。
- “完整导航接口”不存在。
- MuJoCo 内部 goals 还没有作为 ROS API 暴露。

---

## 4. 推荐 Sim2Real 总体架构

```text
                       ┌────────────────────────────┐
                       │        Goal Interface       │
                       │ /goal_pose /waypoints/Nav2 │
                       └──────────────┬─────────────┘
                                      │
                                      v
┌─────────────┐      ┌────────────────────────────┐
│  FAST-LIO   │ ---> │      Goal Manager           │
│  / Odom/TF  │      │ goal -> delta_yaw/cmd_vel   │
└──────┬──────┘      └──────────────┬─────────────┘
       │                            │
       │                            v
       │                 /parkour/goal_yaw
       │                 /cmd_vel or policy cmd
       │
       v
┌─────────────────────┐
│ Real Heightmap Node │  PointCloud2/Map -> 132 floats
└──────────┬──────────┘
           │
           v
 /height_measurements


┌─────────────────────┐
│ Real IMU Bridge     │ sensor_msgs/Imu/Odom -> ang_vel + projected_gravity
└──────────┬──────────┘
           │
           v
 /fast_livo2/state6_imu_prop


┌─────────────────────┐
│ Command Arbiter     │ joy/keyboard/udp/cmd_vel/nav safety arbitration
└──────────┬──────────┘
           │
           v
        deploy_node
           │
           v
      Policy Runner
           │
           v
      Motor Driver
           │
           v
        Real Robot
```

---

## 5. 需要新增的核心模块

## 模块 A：Real Heightmap Node

建议节点名：

```text
real_heightmap_node
```

### 输入

可以根据实际传感器选择：

```text
/sensor/lidar/points        sensor_msgs/PointCloud2
/cloud_registered           sensor_msgs/PointCloud2
/local_elevation_map        grid_map 或 nav_msgs/OccupancyGrid
/tf                         map/odom/base_link/lidar
/odom                       nav_msgs/Odometry
```

### 输出

```text
/height_measurements        std_msgs/Float32MultiArray, size = 132
```

### 功能

1. 获取机器人 base pose。
2. 在 base frame 下生成和训练一致的 12 × 11 采样点。
3. 将采样点变换到 map / odom frame。
4. 从点云或 elevation map 中查询地形高度。
5. 转换为和训练、MuJoCo 一致的 height convention。
6. clamp 到 policy 期望范围，例如 `[-1, 1]`。
7. 对缺失点做 fallback / interpolation。
8. 发布 132 floats。

### 第一版建议

第一版不要直接做复杂全局地图，可以先做：

```text
PointCloud2 + TF -> local 12x11 height grid -> /height_measurements
```

### 验证方法

1. 平地时 heightmap 数值稳定且接近训练中的平地值。
2. 机器人 yaw 旋转时，height grid 跟随 body frame 正确旋转。
3. 台阶 / 障碍物在 RViz 中和 height grid 对齐。
4. 输出范围和 MuJoCo `/height_measurements` 一致。
5. 缺失点比例过高时，触发 safety warning。

---

## 模块 B：Real IMU Bridge

建议节点名：

```text
real_imu_bridge_node
```

### 输入

可选：

```text
/sensor/imu                 sensor_msgs/Imu
/odom                       nav_msgs/Odometry
/fast_lio/odom              nav_msgs/Odometry
/tf                         base_link orientation
```

### 输出

```text
/fast_livo2/state6_imu_prop std_msgs/Float32MultiArray
```

格式：

```text
[wx, wy, wz, gx, gy, gz]
```

其中：

- `wx, wy, wz`：body frame 下角速度。
- `gx, gy, gz`：body frame 下 projected gravity。

### 功能

1. 从 `sensor_msgs/Imu` 读取角速度。
2. 从 orientation 或 TF 计算 projected gravity。
3. 处理 IMU frame 到 base frame 的坐标变换。
4. 做低通滤波。
5. 检查 orientation / angular velocity 是否异常。
6. 超时后通知 safety supervisor。

### 关键风险

IMU 坐标系和训练坐标系不一致会导致 policy 直接异常。因此第一步必须重点验证：

```text
机器人水平站立时 projected gravity 是否接近训练中的水平站立值。
机器人前后俯仰、左右横滚时 gx/gy/gz 符号是否正确。
```

---

## 模块 C：Goal Manager / 目标点接口

建议节点名：

```text
parkour_goal_manager_node
```

当前 `/parkour/goal_yaw` 只是 policy 观测，不是完整目标点接口。建议新增标准目标输入，然后由 goal manager 转成 policy 需要的 yaw。

### MVP 输入

```text
/parkour/goal_pose          geometry_msgs/PoseStamped
/odom 或 /tf                 当前机器人位姿
```

### MVP 输出

```text
/parkour/goal_yaw           std_msgs/Float32MultiArray
/cmd_vel                    geometry_msgs/Twist，可选
```

### 单目标点逻辑

```text
target_in_base = transform(goal_pose, base_link)

distance = sqrt(x^2 + y^2)
delta_yaw = atan2(y, x)

if distance > stop_distance:
    vx = clamp(k_v * distance, 0, max_vx)
    yaw_rate = clamp(k_yaw * delta_yaw, -max_yaw_rate, max_yaw_rate)
else:
    vx = 0
    yaw_rate = 0
    goal_reached = true

goal_yaw = [0, delta_yaw, delta_yaw]
```

### Waypoint / Path 扩展

后续可以支持：

```text
/parkour/waypoints          nav_msgs/Path
```

维护 waypoint queue：

```text
waypoint_0 -> waypoint_1 -> waypoint_2 -> ...
```

每个 tick：

1. 选择当前追踪 waypoint。
2. 距离小于阈值后切换到下一个 waypoint。
3. 计算当前 waypoint 相对 yaw。
4. 计算下一个 waypoint 相对 yaw。
5. 发布：

```text
/parkour/goal_yaw = [0, delta_yaw_to_current, delta_yaw_to_next]
```

这样更贴近当前 policy 里的 `goal_yaw` 输入结构。

### 不建议第一版直接做 Nav2

第一版建议先做：

```text
PoseStamped goal -> goal_manager -> /parkour/goal_yaw + /cmd_vel
```

等以下模块稳定后再接 Nav2：

- 真机 heightmap。
- 真机定位。
- 真机 policy 输入。
- 真机 safety。

---

## 模块 D：CmdVel Bridge / Command Arbiter

建议模块名：

```text
cmd_vel_subscriber
或 command_arbiter
```

当前 deploy 只支持 keyboard、UDP、joy，不支持 `/cmd_vel`。

### 输入

```text
/joy
/keyboard internal
/udp command
/cmd_vel                    geometry_msgs/Twist
/emergency_stop             可选
/navigation_active          可选
```

### 输出

给 policy 的：

```text
vx, vy, yaw_rate
```

### 推荐优先级

```text
E-stop
> joint damping / emergency state
> joy manual override
> keyboard
> UDP
> nav / cmd_vel
> idle
```

### 必须有的安全逻辑

1. `/cmd_vel` 超时自动归零。
2. joy 有输入时抢占自动导航。
3. goal reached 后自动停止。
4. 定位 / heightmap / IMU 超时后自动停止或切 damping。
5. 限制 `vx`、`vy`、`yaw_rate`。
6. 对命令做一阶滤波，避免跳变。

### 第一版实现方式

如果不想大改 `deploy_node`，可以先在 `deploy_node` 里加一个 `/cmd_vel` subscriber，并把它作为最低优先级 command source。

---

## 模块 E：Safety Supervisor

真机 sim2real 必须有独立或半独立安全监控。

建议模块名：

```text
safety_supervisor
```

### 监控项

1. IMU 超时。
2. heightmap 超时。
3. motor feedback 超时。
4. joint position 超限。
5. joint velocity 超限。
6. torque 超限。
7. motor temperature 超限。
8. motor error code。
9. base roll / pitch 过大。
10. command timeout。
11. policy output NaN / Inf。
12. action delta 过大。
13. heightmap 缺失点比例过高。
14. localization lost。
15. emergency stop 输入。

### 状态建议

```text
NORMAL      -> 正常 RL
WARNING     -> 限速 / 限动作幅度
DANGER      -> 切 JOINT_DAMPING
CRITICAL    -> zero torque 或 emergency stop
```

### 初期建议

第一版可以先放在 `deploy_node` 内部，后续再抽象成独立类或节点。

---

## 模块 F：Logging / Replay / Debug

真机调试强烈建议加日志与回放。

### 需要记录的数据

```text
timestamp
state machine state
commands vx/vy/yaw_rate
goal_yaw
imu angular velocity
imu projected gravity
heightmap 132 floats
joint q/dq/tau/temp/error
policy action raw
policy action clipped
target joint position
motor command
safety state
sensor timeout state
```

### 输出形式

可以选择：

```text
rosbag2
CSV
JSONL
binary log
```

### 建议新增工具

```text
replay_policy_node
```

用真实日志里的：

```text
imu + joint state + heightmap + goal_yaw + commands
```

离线重跑 policy，定位某一帧动作异常的原因。

---

## 6. 推荐新增文件

第一轮建议新增：

```text
deploy_cpp/include/cmd_vel_subscriber.h
deploy_cpp/src/cmd_vel_subscriber.cpp

deploy_cpp/include/goal_manager.h
deploy_cpp/src/goal_manager.cpp

deploy_cpp/src/real_imu_bridge_node.cpp

deploy_cpp/src/real_heightmap_node.cpp

deploy_cpp/launch/real_deploy.launch.py

deploy_cpp/config/robots/mybot_v3_parkour_real.yaml
```

后续可选：

```text
deploy_cpp/include/safety_supervisor.h
deploy_cpp/src/safety_supervisor.cpp

deploy_cpp/include/deploy_logger.h
deploy_cpp/src/deploy_logger.cpp

deploy_cpp/src/replay_policy_node.cpp
```

---

## 7. 推荐配置拆分

建议将 sim 和 real config 分开：

```text
deploy_cpp/config/robots/mybot_v3_parkour_sim.yaml
deploy_cpp/config/robots/mybot_v3_parkour_real.yaml
```

real config 可新增：

```yaml
topics:
  imu: /fast_livo2/state6_imu_prop
  height: /height_measurements
  goal_yaw: /parkour/goal_yaw
  cmd_vel: /cmd_vel
  goal_pose: /parkour/goal_pose
  odom: /odom
  pointcloud: /cloud_registered

safety:
  imu_timeout_s: 0.1
  height_timeout_s: 0.2
  cmd_timeout_s: 0.3
  motor_feedback_timeout_s: 0.05
  max_roll_deg: 35.0
  max_pitch_deg: 35.0
  max_joint_vel: 30.0
  max_action_delta: 0.25

navigation:
  stop_distance: 0.25
  waypoint_reach_distance: 0.35
  max_vx: 0.4
  max_vy: 0.0
  max_yaw_rate: 0.8
  yaw_kp: 1.5
  distance_kp: 0.5

heightmap:
  source: pointcloud
  frame_id: base_link
  map_frame: odom
  pointcloud_topic: /cloud_registered
  publish_topic: /height_measurements
  num_x: 12
  num_y: 11
  min_height: -1.0
  max_height: 1.0
  missing_value: 0.0
```

---

## 8. 推荐实现顺序

### 阶段 1：真机低风险 Bringup

目标：不开 RL 或只做单步 RL，先确认硬件和状态输入。

任务：

1. 检查 motor mapping。
2. 检查 encoder offset。
3. 验证 damping 模式。
4. 验证 standup。
5. 验证 `/joint_states` 可视化。
6. 接入真实 IMU bridge。
7. 验证 projected gravity 符号和坐标系。

---

### 阶段 2：Policy 输入打通

目标：policy 输入都来自真机 producer，但先低速或 single-step。

任务：

1. `/fast_livo2/state6_imu_prop` 正常。
2. `/height_measurements` 正常。
3. `/parkour/goal_yaw` 正常。
4. command source 正常。
5. policy input logging 正常。
6. single-step RL debug 正常。

---

### 阶段 3：真实 Heightmap

目标：真机用 LiDAR / 点云生成 132 点 heightmap。

第一版：

```text
PointCloud2 -> base frame local grid -> /height_measurements
```

验证：

1. 平地数值稳定。
2. 障碍物位置和 RViz 一致。
3. base yaw 变化时 grid 正确旋转。
4. 缺失点处理合理。
5. 输出范围和 MuJoCo 一致。

---

### 阶段 4：目标点接口 MVP

目标：能发一个目标点，让机器人朝目标走。

新增：

```text
/parkour/goal_pose geometry_msgs/PoseStamped
parkour_goal_manager_node
/cmd_vel bridge
```

先做：

```text
目标点 -> delta_yaw -> vx/yaw_rate
```

---

### 阶段 5：Waypoint / Path

目标：支持多个目标点。

新增：

```text
/parkour/waypoints nav_msgs/Path
```

输出：

```text
/parkour/goal_yaw = [0, delta_yaw_to_current, delta_yaw_to_next]
```

---

### 阶段 6：Nav2 或高层规划

最后再接：

```text
Nav2 NavigateToPose
Nav2 FollowPath
/cmd_vel
```

或者自研轻量 planner。

---

## 9. 最小 MVP 清单

如果只做 sim2real 第一版，最少建议补：

| 优先级 | 模块 | 作用 |
|---|---|---|
| P0 | Safety Supervisor | 真机保护，超时 / 跌倒 / 电机异常进 damping |
| P0 | Real IMU Bridge | `sensor_msgs/Imu` / Odom 转 6 维 policy IMU |
| P0 | Real Heightmap Node | 点云 / 地图转 132 维 `/height_measurements` |
| P1 | Goal Manager | `/goal_pose` 或 `/waypoints` 转 `/parkour/goal_yaw` |
| P1 | CmdVel Bridge / Command Arbiter | 自动导航速度接入 policy command |
| P1 | Logging / Replay | 真机调试和问题复盘 |

---

## 10. 第一版目标点接口建议

第一版建议保留现有 `/parkour/goal_yaw`，不要改 policy 输入。

新增输入：

```text
/parkour/goal_pose geometry_msgs/PoseStamped
```

新增节点：

```text
parkour_goal_manager_node
```

输出：

```text
/parkour/goal_yaw std_msgs/Float32MultiArray
/cmd_vel geometry_msgs/Twist，可选
```

单目标点时：

```text
/parkour/goal_yaw = [0, delta_yaw, delta_yaw]
```

多 waypoint 时：

```text
/parkour/goal_yaw = [0, delta_yaw_to_current, delta_yaw_to_next]
```

这样对当前代码侵入最小，同时能把后续 Nav2 或自研 planner 接进来。

---

## 11. 下一步建议

建议下一步先实现三个最小模块：

1. `real_imu_bridge_node`
2. `cmd_vel_subscriber` / `command_arbiter`
3. `parkour_goal_manager_node`

然后再做最难的：

```text
real_heightmap_node
```

原因：heightmap 是 sim2real 中最容易出数值 convention 和坐标系问题的部分，建议在 IMU、目标点、command、safety 都稳定后重点调。第一版可以先用固定平地 heightmap 或录制的 sim heightmap 做 pipeline 验证，再切真实点云。
