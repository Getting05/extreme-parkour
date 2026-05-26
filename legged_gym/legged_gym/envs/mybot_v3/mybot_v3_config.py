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
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
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
        stiffness = {"joint": 40.0}  # [N*m/rad]
        damping = {"joint": 1}  # [N*m*s/rad]
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

    class rewards:
        class scales:
            # tracking rewards
            tracking_goal_vel = 1.8
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.0e-7
            collision = -5.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.000001
            hip_pos = -1.5
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = .9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.6   
        base_height_target = 0.26
        max_contact_force = 40. # forces above this value are penalized





class MybotV3RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_mybot_v3"
