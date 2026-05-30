/**
 * @file robot_runtime_config.h
 * @brief Runtime configuration loaded from YAML for parkour deployment.
 */
#pragma once

#include <array>
#include <string>
#include <vector>

#include "robot_config.h"

namespace deploy {

struct MotorMapping {
  int motor_id = 0;
  int port_idx = 0;
  bool is_reversed = false;
};

struct RobotRuntimeConfig {
  int num_of_dofs = NUM_JOINTS;
  float dt = 0.005f;
  int decimation = 4;

  std::array<float, NUM_JOINTS> kp_joint{};
  std::array<float, NUM_JOINTS> kd_joint{};
  std::array<float, NUM_JOINTS> torque_limits{};

  std::array<float, NUM_JOINTS> default_dof_pos{};
  std::array<float, NUM_JOINTS> standup_target_pos{};
  std::array<float, NUM_JOINTS> policy_dof_pos{};
  std::array<float, NUM_JOINTS> joint_pos_lower{};
  std::array<float, NUM_JOINTS> joint_pos_upper{};

  std::array<std::string, NUM_JOINTS> joint_names{};
  std::array<std::string, NUM_JOINTS> joint_controller_names{};

  std::array<int, NUM_JOINTS> joint_mapping{};
  std::array<bool, NUM_JOINTS> motor_is_reversed{};
  std::array<MotorMapping, NUM_JOINTS> motor_map{};
  std::array<float, NUM_JOINTS> joint_transmission_ratio{};
  std::vector<int> wheel_indices;

  // Model paths (Approach B: separate sub-components)
  std::string heightmap_encoder_path = "policy/heightmap_encoder.jit";
  std::string history_encoder_path = "policy/history_encoder.jit";
  std::string actor_backbone_path = "policy/actor_backbone.jit";
  std::string device = "cpu";

  std::string port0 = "/dev/ttyUSB0";
  std::string port1 = "/dev/ttyUSB1";
  std::string imu_topic = "/fast_livo2/state6_imu_prop";
  std::string height_topic = "/height_measurements";
  float imu_yaw_correction_deg = 0.0f;

  std::string robot_name = "mybot_v3_parkour";
  std::string urdf_relpath = "robot/mybot_v3/urdf/mybot_v3.urdf";
  std::string mujoco_xml_relpath = "robot/mybot_v3/xml/mybot_v3.xml";
  std::string isaac_xml_relpath = "robot/mybot_v3/xml/mybot_v3.xml";

  // Observation normalization scales (from training config)
  float ang_vel_scale = 0.25f;
  float dof_pos_scale = 1.0f;
  float dof_vel_scale = 0.05f;
  float command_scale = 1.0f;      // commands[:, 0] is passed directly
  float action_scale = 0.25f;
  float yaw_scale = 1.5f;          // yaw estimate is scaled by 1.5

  // Height observation
  float height_bias = 0.3f;        // root_z - 0.3 - measured_heights in training
  float nominal_base_height = 0.34f;
  float height_measurement_scale = 1.0f;
  float height_measurement_offset = 0.0f;

  // Safety checks
  bool require_imu_ready_for_rl = true;
  bool height_sanity_check_enable = true;
  float gravity_norm_min = 0.8f;
  float gravity_norm_max = 1.2f;
  float gravity_z_max = -0.3f;
  float height_distance_min = -1.5f;
  float height_distance_max = 1.5f;

  float cmd_deadband = 0.05f;
  float control_dt = 0.02f;
  float standup_duration = 2.0f;

  float cmd_vx_min = -0.6f;
  float cmd_vx_max = 1.5f;
  float cmd_vy_min = -0.6f;
  float cmd_vy_max = 0.6f;
  float cmd_yaw_min = -1.0f;
  float cmd_yaw_max = 1.0f;
  float cmd_vx_step = 0.1f;
  float cmd_vy_step = 0.1f;
  float cmd_yaw_step = 0.2f;

  float clip_obs = 100.0f;
  float clip_actions = 1.2f;      // from training: normalization.clip_actions
  float kd_damp_motor = 0.1f;
  std::array<int, 4> hip_indices = {0, 3, 6, 9};

  bool teleop_udp_enable = false;
  int teleop_udp_port = 9870;
  bool joy_enable = true;
  std::string joy_topic = "/joy";
  float joy_axis_deadzone = 0.15f;
  float joy_timeout_s = 0.5f;
  int joy_axis_vx = 1;
  int joy_axis_vy = 0;
  int joy_axis_yaw = 3;
  bool joy_invert_vx = true;
  bool joy_invert_vy = false;
  bool joy_invert_yaw = false;
  int joy_button_stand_up = 0;
  int joy_button_return_default = 1;
  int joy_button_rl = 2;
  int joy_button_damping = 3;
  int joy_button_single_step = 4;
  int joy_button_joint_sweep = 5;
  int joy_button_idle = 6;
  int joy_button_confirm = 7;
  int joy_button_emergency = 8;
  bool debug_print_policy = true;
  int debug_print_interval = 50;
};

RobotRuntimeConfig default_robot_runtime_config();
RobotRuntimeConfig load_robot_runtime_config(const std::string &yaml_file);

} // namespace deploy
