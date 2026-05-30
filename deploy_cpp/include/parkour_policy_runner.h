/**
 * @file parkour_policy_runner.h
 * @brief Extreme-parkour heightmap student policy inference.
 *
 * Three JIT sub-models (Approach B):
 *   1. heightmap_encoder: HeightmapEncoder(backbone+GRU+output_mlp)
 *      Input:  heightmap(1,132) + proprio(1,49) + hidden(1,1,512)
 *      Output: latent_yaw(1,34) + new_hidden(1,1,512)
 *   2. history_encoder: StateHistoryEncoder
 *      Input:  history(1,10,53)
 *      Output: latent(1,20)
 *   3. actor_backbone: Sequential MLP
 *      Input:  (1,114) = [prop(53) + scan_latent(32) + priv_explicit(9) +
 * hist_latent(20)] Output: actions(1,12)
 */
#pragma once

#include <array>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

#include "robot_runtime_config.h"

namespace deploy {

class ParkourPolicyRunner {
public:
  explicit ParkourPolicyRunner(const RobotRuntimeConfig &config);

  void reset();

  /**
   * @brief Run one inference step.
   * @param commands       [vx, vy, yaw_rate] velocity commands
   * @param ang_vel        [wx, wy, wz] body angular velocity from IMU
   * @param projected_gravity [gx, gy, gz] projected gravity from IMU
   * @param dof_pos        Current joint positions (URDF order)
   * @param dof_vel        Current joint velocities (URDF order)
   * @param height_measurements  132 height points from LiDAR
   * @param target_dof_pos [out] Target joint positions for PD control
   * @param actions        [out] Raw policy actions
   */
  void step(const std::array<float, 3> &commands,
            const std::array<float, 3> &ang_vel,
            const std::array<float, 3> &projected_gravity,
            const std::array<float, NUM_JOINTS> &dof_pos,
            const std::array<float, NUM_JOINTS> &dof_vel,
            const std::array<float, NUM_HEIGHT_POINTS> &height_measurements,
            std::array<float, NUM_JOINTS> &target_dof_pos,
            std::array<float, NUM_ACTIONS> &actions);

private:
  torch::Tensor
  build_proprio_student(const std::array<float, 3> &commands,
                        const std::array<float, 3> &ang_vel,
                        const std::array<float, 3> &projected_gravity,
                        const std::array<float, NUM_JOINTS> &dof_pos,
                        const std::array<float, NUM_JOINTS> &dof_vel);

  torch::Tensor build_proprio_full(const torch::Tensor &proprio_student,
                                   const torch::Tensor &yaw_estimate);

  void update_obs_history(const torch::Tensor &proprio_full);

  std::array<float, NUM_JOINTS>
  target_from_actions(const std::array<float, NUM_ACTIONS> &actions) const;

  void validate_model_shapes();

  RobotRuntimeConfig config_;
  torch::Device device_;

  // JIT models
  torch::jit::script::Module heightmap_encoder_;
  torch::jit::script::Module history_encoder_;
  torch::jit::script::Module actor_backbone_;

  // GRU hidden state for heightmap_encoder
  torch::Tensor gru_hidden_; // (1, 1, 512)

  // Observation history buffer for history_encoder
  torch::Tensor obs_history_; // (1, HISTORY_LEN, N_PROPRIO) = (1, 10, 53)

  // Last actions for proprio computation
  torch::Tensor last_actions_; // (1, NUM_ACTIONS)

  // Default dof pos tensor
  torch::Tensor default_dof_pos_; // (1, NUM_JOINTS)

  uint64_t infer_count_ = 0;
};

} // namespace deploy
