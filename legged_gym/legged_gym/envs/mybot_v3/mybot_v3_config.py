# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# 关节顺序就是 URDF 里 12 个转动关节出现的顺序
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MybotV3RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        reorder_dofs = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 30.0}  # [N*m/rad]
        damping = {"joint": 0.6}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["body"]
        self_collisions = 1
        collapse_fixed_joints = False
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35


class MybotV3RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_mybot_v3"
