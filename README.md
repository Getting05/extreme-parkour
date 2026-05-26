# Extreme Parkour with Legged Robots #
<p align="center">
<img src="./images/teaser.jpeg" width="80%"/>
</p>

**Authors**: [Xuxin Cheng*](https://chengxuxin.github.io/), [Kexin Shi*](https://tenhearts.github.io/), [Ananye Agarwal](https://anag.me/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)  
**Website**: https://extreme-parkour.github.io  
**Paper**: https://arxiv.org/abs/2309.14341  
**Tweet Summary**: https://twitter.com/pathak2206/status/1706696237703901439

### Installation ###
```bash
conda create -n parkour python=3.8
conda activate parkour
cd
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:chengxuxin/extreme-parkour.git
cd extreme-parkour
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
# Originally trained with Preview3, but haven't seen bugs using Preview4.
cd isaacgym/python && pip install -e .
cd ~/extreme-parkour/rsl_rl && pip install -e .
cd ~/extreme-parkour/legged_gym && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###
`cd legged_gym/scripts`
1. Train base policy:  
```bash
python train.py --exptid xxx-xx-WHATEVER --device cuda:0
```
Train 10-15k iterations (8-10 hours on 3090) (at least 15k recommended).

2. Train distillation policy:
```bash
python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera
```
Train 5-10k iterations (5-10 hours on 3090) (at least 5k recommended). 
>You can run either base or distillation policy at arbitary gpu # as long as you set `--device cuda:#`, no need to set `CUDA_VISIBLE_DEVICES`.

3. Play base policy:
```bash
python play.py --exptid xxx-xx
```
No need to write the full exptid. The parser will auto match runs with first 6 strings (xxx-xx). So better make sure you don't reuse xxx-xx. Delay is added after 8k iters. If you want to play after 8k, add `--delay`

4. Play distillation policy:
```bash
python play.py --exptid yyy-yy --delay --use_camera
```

5. Save models for deployment:
```bash
python save_jit.py --exptid xxx-xx
```
This will save the models in `legged_gym/logs/parkour_new/xxx-xx/traced/`.

### Viewer Usage
Can be used in both IsaacGym and web viewer.
- `ALT + Mouse Left + Drag Mouse`: move view.
- `[ ]`: switch to next/prev robot.
- `Space`: pause/unpause.
- `F`: switch between free camera and following camera.

### Arguments
- --exptid: string, can be `xxx-xx-WHATEVER`, `xxx-xx` is typically numbers only. `WHATEVER` is the description of the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --delay: whether add delay or not.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --use_camera: use camera or scandots.
- --web: used for playing on headless machines. It will forward a port with vscode and you can visualize seemlessly in vscode with your idle gpu or cpu. [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) vscode extension required, otherwise you can view it in any browser.

### Acknowledgement
https://github.com/leggedrobotics/legged_gym  
https://github.com/Toni-SM/skrl

---

## LiDAR Heightmap Distillation (自研机器人扩展)

本项目在原始 Extreme Parkour 的基础上，新增了**第二阶段 LiDAR 高程图蒸馏**方法，适用于：
- 没有足端接触传感器的自研四足机器人
- 使用 LiDAR 精确定位 + 建图获取周围地形高程信息（替代深度相机）

### 改动概述

| 原方法 (depth distillation) | 新方法 (heightmap distillation) |
|---|---|
| 深度相机 58x87 图像输入 | LiDAR 高程图 N 点 1D 向量输入 |
| CNN + GRU 编码器 | MLP + GRU 编码器 |
| 保留 foot_contacts (4维) | 去除 foot_contacts（置零屏蔽） |
| `learn_vision` 训练路径 | `learn_heightmap` 训练路径 |

### 架构设计

```
教师 (Teacher):
  obs(53维 proprio + 132 scan + priv) → scan_encoder → actor → actions

学生 (Student - Heightmap):
  LiDAR heightmap(132点) + proprio_student(49维, 无foot_contacts)
    → HeightmapMLPBackbone(132 → 128 → 64 → 32)
    → HeightmapEncoder(MLP + GRU → 32维 terrain_latent + 2维 yaw)
  obs_student(53维, foot_contacts置零, yaw替换) + terrain_latent
    → heightmap_actor → actions

蒸馏损失:
  L_action = ||actions_student - actions_teacher||₂
  L_yaw    = ||yaw_student - yaw_teacher||₂
  L_latent = ||heightmap_latent - scandots_latent_teacher||₂
```

### 使用教程

#### 1. 配置你的机器人

编辑 `legged_gym/legged_gym/envs/mybot_v3/mybot_v3_config.py`：

```python
class MybotV3RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        include_foot_contacts = False  # 无足端接触传感器

    class heightmap(LeggedRobotCfg.heightmap):
        use_heightmap = True           # 启用LiDAR高程蒸馏
        n_points = 132                 # 高程采样点数
        noise_std = 0.02               # LiDAR测量噪声 [m]

class MybotV3RoughCfgPPO(LeggedRobotCfgPPO):
    class heightmap_encoder:
        if_heightmap = True
        n_points = 132
        n_proprio_student = 49         # 53 - 4 (去掉foot_contacts)
        backbone_hidden_dims = [128, 64]
        backbone_output_dim = 32
        hidden_size = 512
        output_dim = 32
        learning_rate = 1.e-3
        num_steps_per_env = 120        # update_interval * 24
        latent_loss_weight = 1.0
        action_loss_weight = 1.0
        yaw_loss_weight = 1.0
```

#### 2. 训练第一阶段教师模型（不变）

```bash
cd legged_gym/scripts
python train.py --task mybot_v3 --exptid 001-01-teacher --device cuda:0
```

训练 10-15k iterations。教师模型使用完整的特权信息（含 foot_contacts + 特权高程扫描）。

#### 3. 训练第二阶段 Heightmap 蒸馏

```bash
python train.py --task mybot_v3 --exptid 002-01-heightmap-student --device cuda:0 \
    --resume --resumeid 001-01
```

当配置中 `heightmap.use_heightmap = True` 时，runner 自动选择 `learn_heightmap` 训练路径。

训练 5-10k iterations。学生模型：
- 从教师 actor 权重初始化 `heightmap_actor`
- `heightmap_encoder` 从零开始训练
- foot_contacts 在 student obs 中被置零（当前帧 + 10帧历史）

#### 4. 播放/评估蒸馏策略

```bash
python play.py --task mybot_v3 --exptid 002-01
```

#### 5. 关键参数调节

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `n_points` | 高程采样点数 | 132（与教师一致）|
| `noise_std` | 高程噪声标准差 | 0.01-0.05 |
| `hidden_size` | GRU 隐藏层大小 | 512 |
| `learning_rate` | 学习率 | 1e-3 |
| `latent_loss_weight` | latent 蒸馏损失权重 | 0.5-2.0 |
| `action_loss_weight` | 动作蒸馏损失权重 | 1.0 |
| `yaw_loss_weight` | yaw 估计损失权重 | 0.5-1.0 |

#### 6. 部署注意事项

部署时，学生策略的输入为：
- **Proprioception (49维)**: IMU + 关节角度/速度/上一步动作（无 foot_contacts）
- **LiDAR Heightmap (132点)**: 机器人周围的相对高程采样

推理流程：
```python
# 1. 获取传感器数据
proprio = get_proprio_without_foot_contacts()  # 49维
heightmap = get_lidar_heightmap()              # 132点

# 2. Heightmap encoder 前向
latent_and_yaw = heightmap_encoder(heightmap, proprio)
terrain_latent = latent_and_yaw[:, :-2]        # 32维
yaw_estimate = 1.5 * latent_and_yaw[:, -2:]   # 2维

# 3. 构造 actor 输入 (53维 proprio, foot_contacts位置填零)
obs_full = build_obs_with_zero_foot_contacts(proprio, yaw_estimate)

# 4. Actor 前向
action = heightmap_actor(obs_full, hist_encoding=True, scandots_latent=terrain_latent)
```

### 文件修改清单

| 文件 | 改动 |
|------|------|
| `rsl_rl/modules/heightmap_backbone.py` | **新建** HeightmapMLPBackbone + HeightmapEncoder |
| `rsl_rl/modules/__init__.py` | 添加 heightmap 导入 |
| `rsl_rl/algorithms/ppo.py` | 新增 heightmap 初始化 + `update_heightmap_actor()` |
| `rsl_rl/runners/on_policy_runner.py` | 新增 `learn_heightmap()` + save/load/inference |
| `legged_gym/envs/base/legged_robot_config.py` | 新增 heightmap 配置类 |
| `legged_gym/envs/base/legged_robot.py` | 新增 `get_noisy_heightmap()` |
| `legged_gym/envs/mybot_v3/mybot_v3_config.py` | 启用 heightmap + 关闭 foot_contacts |

---

### Citation
If you found any part of this code useful, please consider citing:
```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```