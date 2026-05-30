# Extreme-Parkour Heightmap Student 部署框架

基于 `reference/deploy_cpp` 的 CSE 部署框架改写，用于部署修改后的 extreme-parkour 学生模型（去掉足端接触传感器和深度相机，保留 LiDAR 高程图观测）。

## 与参考框架的主要区别

| 项目 | 参考 CSE 框架 | 本框架 (Parkour) |
|------|-------------|-----------------|
| 策略模型 | `adaptation_module.jit` + `body.jit` | `heightmap_encoder.jit` + `history_encoder.jit` + `actor_backbone.jit` |
| 编码器 | MLP estimator（无时序） | HeightmapEncoder（MLP backbone + GRU，有时序隐状态） |
| 本体感知维度 | 53（含足端接触） | 49（学生端）/ 53（全量，足端置零） |
| 高程点数 | 77（11×7） | 132（12×11） |
| 是否需要 IMU 角速度 | 否（仅 gravity） | 是（ang_vel 参与 proprio 构建） |
| Actor 输入 | 观测历史 + latent | proprio(53) + terrain_latent(32) + priv_explicit(9) + hist_latent(20) = 114 |

## 目录结构

```
deploy_cpp/
├── CMakeLists.txt                     # ROS2 ament_cmake 构建
├── package.xml                        # ROS2 包描述
├── README.md                          # 本文档
│
├── include/
│   ├── robot_config.h                 # [新] 编译期维度常量（49/53/132/114/753等）
│   ├── robot_runtime_config.h         # [改] 运行时配置结构体，模型路径改为三个子模块
│   ├── parkour_policy_runner.h        # [新] Parkour 策略推理器头文件
│   ├── imu_subscriber.h               # [复用] IMU 订阅
│   ├── height_subscriber.h            # [复用] 高程图订阅（自动适配 132 点）
│   ├── motor_driver.h                 # [复用] 电机驱动
│   ├── keyboard_controller.h          # [复用] 键盘控制
│   ├── state_machine.h                # [复用] 状态机
│   ├── udp_controller.h               # [复用] UDP 遥操作
│   ├── robot_visualizer.h             # [复用] RViz 可视化
│   └── Unitree_Motor/                 # [复用] Unitree 电机 SDK 头文件
│
├── src/
│   ├── deploy_node.cpp                # [改] 主节点，CSEPolicyRunner → ParkourPolicyRunner
│   ├── parkour_policy_runner.cpp      # [新] 三模型推理管线实现
│   ├── robot_runtime_config.cpp       # [改] YAML 加载器，适配新模型路径
│   ├── imu_subscriber.cpp             # [复用]
│   ├── height_subscriber.cpp          # [复用]
│   ├── motor_driver.cpp               # [复用]
│   ├── keyboard_controller.cpp        # [复用]
│   ├── state_machine.cpp              # [复用]
│   ├── udp_controller.cpp             # [复用]
│   ├── robot_visualizer.cpp           # [复用]
│   └── motor_debug_node.cpp           # [复用] 电机调试节点
│
├── config/robots/
│   └── mybot_v3_parkour.yaml          # [新] mybot_v3 训练参数匹配的部署配置
│
├── launch/
│   └── deploy.launch.py               # [改] 默认加载 mybot_v3_parkour.yaml
│
├── scripts/
│   └── export_heightmap_jit.py        # [新] 从训练 checkpoint 导出 JIT 模型
│
├── policy/                            # JIT 模型存放目录（由导出脚本生成）
│   ├── heightmap_encoder.jit
│   ├── history_encoder.jit
│   └── actor_backbone.jit
│
├── lib/                               # Unitree 电机 SDK 动态库
│   ├── libUnitreeMotorSDK_Arm64.so
│   └── libUnitreeMotorSDK_Linux64.so
│
├── robot/                             # URDF / MuJoCo XML 模型
│   └── mybot_v3/
│
└── reference/                         # 原始参考框架（不参与编译）
    └── deploy_cpp/
```

## 观测空间映射

### 学生本体感知（49 维，无足端接触）

```
索引        内容                            缩放
[0:3]      base_ang_vel                   × 0.25
[3:5]      roll, pitch                    从 projected_gravity 计算
[5]        0.0                            masked
[6:8]      delta_yaw, delta_next_yaw      由 heightmap_encoder 估计，× 1.5
[8:10]     0.0, 0.0                       masked commands
[10]       cmd_vx                         × command_scale
[11]       1.0                            非gap地形标志
[12]       0.0                            gap地形标志
[13:25]    (dof_pos - default)            × dof_pos_scale (1.0)
[25:37]    dof_vel                        × dof_vel_scale (0.05)
[37:49]    last_actions                   —
```

### Actor Backbone 输入（114 维）

```
[0:53]     proprio_full（49 + 4 零值足端）
[53:85]    terrain_latent（32 维，来自 heightmap_encoder）
[85:94]    priv_explicit（9 维，部署时置零）
[94:114]   hist_latent（20 维，来自 history_encoder）
```

## 推理管线

每个控制步（20ms）执行以下流程：

1. **构建 proprio_student（49 维）**：从 IMU、电机反馈、指令构建
2. **HeightmapEncoder 推理**：输入 heightmap(132) + proprio(49) + GRU hidden → 输出 terrain_latent(32) + yaw(2)
3. **构建 proprio_full（53 维）**：插入 yaw 估计到 [6:8]，足端接触 [49:53] 置零
4. **更新观测历史**：将当前 proprio 追加到 10 帧历史缓冲
5. **HistoryEncoder 推理**：输入 history(10×53) → hist_latent(20)
6. **ActorBackbone 推理**：输入 cat(proprio_53, terrain_32, zeros_9, hist_20) = 114 → actions(12)
7. **目标位置计算**：`target = default_dof_pos + actions × action_scale`

## 使用方法

### 1. 导出 JIT 模型

```bash
conda activate parkour
python deploy_cpp/scripts/export_heightmap_jit.py \
    --checkpoint legged_gym/logs/parkour_new/student/model_34000.pt \
    --output_dir deploy_cpp/policy
```

输出三个 JIT 文件：
- `heightmap_encoder.jit` — HeightmapEncoder (backbone + GRU + output_mlp)
- `history_encoder.jit` — StateHistoryEncoder (Conv1D)
- `actor_backbone.jit` — Actor MLP (114 → 12)

### 2. 编译

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select deploy_cpp \
    --cmake-args -DCMAKE_PREFIX_PATH=/opt/libtorch
```

### 3. 运行

```bash
source install/setup.bash

# 真机部署
ros2 launch deploy_cpp deploy.launch.py

# 无电机调试（测试推理管线，无需电机硬件）
ros2 launch deploy_cpp deploy.launch.py debug_no_motor:=true

# 指定配置文件
ros2 launch deploy_cpp deploy.launch.py \
    robot_config_file:=/path/to/custom_config.yaml
```

### 4. MuJoCo 仿真

MuJoCo 仿真环境通过 `sim/mujoco_sim_node.py` 提供，它会启动一个 MuJoCo 物理仿真窗口，并通过 ROS2 topic 与 `deploy_node` 通信。

> **⚠️ 键盘控制注意事项**：`ros2 launch` 不会转发 stdin 给子进程，因此**键盘按键（0-6, W/S 等）在 launch 模式下无效**。如果需要键盘控制，请使用下面的"分别启动"方式，用 `ros2 run` 直接启动 deploy_node。

#### 方式一：分别启动（推荐，支持键盘控制）

终端 1 — MuJoCo 仿真节点：
```bash
conda activate mujoco_sim
source /opt/ros/humble/setup.bash
python3 deploy_cpp/sim/mujoco_sim_node.py \
    --robot-config deploy_cpp/config/robots/mybot_v3_parkour.yaml
```

终端 2 — 部署控制节点（直接 `ros2 run`，键盘可用）：
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run deploy_cpp deploy_node --ros-args \
    -p robot_config_file:=$(pwd)/deploy_cpp/config/robots/mybot_v3_parkour.yaml \
    -p sim_mode:=true
```

此时在终端 2 中按数字键即可切换状态（1=站立, 2=RL, W/S=前进后退 等）。

#### 方式二：一键 launch（仅限手柄/UDP 控制）

自动启动 MuJoCo 仿真 + deploy_node，但键盘输入无效，需搭配手柄或 UDP 遥控使用：

```bash
conda activate mujoco_sim
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch deploy_cpp sim.launch.py

# 启用 pingpong 模式（仿真与控制严格同步）
ros2 launch deploy_cpp sim.launch.py sim_pingpong_mode:=true

# 手柄控制（另开一个终端）
ros2 run joy joy_node
```

#### MuJoCo 仿真 ROS2 Topic 架构

```
mujoco_sim_node                          deploy_node
┌──────────────┐                    ┌──────────────┐
│  MuJoCo      │  /mujoco/joint_state   │  Policy      │
│  Physics     │ ──────────────────────→ │  Inference   │
│              │  /fast_livo2/state6_imu │              │
│  PD Control  │ ──────────────────────→ │  Observation │
│              │  /height_measurements   │  Builder     │
│  Viewer      │ ──────────────────────→ │              │
│              │                         │              │
│              │  /mujoco/joint_cmd      │  PD Target   │
│              │ ←────────────────────── │  Output      │
└──────────────┘                    └──────────────┘
```

| Topic | 方向 | 内容 |
|-------|------|------|
| `/mujoco/joint_state` | sim → deploy | 24 floats: 12 pos + 12 vel |
| `/fast_livo2/state6_imu_prop` | sim → deploy | 6 floats: 3 ang_vel + 3 proj_gravity |
| `/height_measurements` | sim → deploy | 132 floats: 12×11 高程距离 |
| `/mujoco/joint_cmd` | deploy → sim | 36 floats: 12 target + 12 kp + 12 kd |

#### 仿真依赖

```bash
conda activate mujoco_sim
pip install mujoco numpy pyyaml
# rclpy 需要通过 ROS2 环境提供
```

### 5. 控制

| 按键 | 功能 |
|------|------|
| `0` | IDLE（零力矩） |
| `1` | STAND_UP（站立插值） |
| `2` | RL（策略控制） |
| `3` | JOINT_DAMPING（被动阻尼） |
| `4` | RETURN_DEFAULT（回默认位姿） |
| `5` | SINGLE_STEP_RL（单步确认） |
| `6` | JOINT_SWEEP（关节扫描调试） |
| `W/S` | 前进/后退速度 |
| `Q/E` | 横向速度 |
| `A/D` | 偏航速率 |
| `R` | 速度归零 |
| `Space` | 紧急停止 |
| `Esc` | 退出 |

手柄控制：运行 `ros2 run joy joy_node`，通过 `/joy` topic 控制。

## 配置说明

配置文件位于 `config/robots/mybot_v3_parkour.yaml`，关键参数：

```yaml
# 控制频率
dt: 0.005              # 仿真步长（与训练一致）
decimation: 4          # 每 4 步执行一次策略
control_dt: 0.02       # 实际控制周期 = dt × decimation = 20ms

# PD 增益（与训练一致）
kp_joint: [35, ...]    # 位置增益
kd_joint: [1, ...]     # 速度增益

# 观测缩放（与训练一致）
ang_vel_scale: 0.25
dof_vel_scale: 0.05
action_scale: 0.25
yaw_scale: 1.5         # yaw 估计缩放系数
clip_actions: 1.2       # 动作裁剪范围

# 模型路径
heightmap_encoder_path: policy/heightmap_encoder.jit
history_encoder_path: policy/history_encoder.jit
actor_backbone_path: policy/actor_backbone.jit
```

## 依赖

- ROS2 Humble
- LibTorch (C++ PyTorch)
- yaml-cpp
- Unitree Motor SDK (GO-M8010-6)
