/**
 * @file parkour_policy_runner.cpp
 * @brief Extreme-parkour heightmap student policy runner implementation.
 *
 * Inference pipeline per step:
 *   1. Build proprio_student (49d) from sensor data
 *   2. Call heightmap_encoder(heightmap, proprio_student, gru_hidden)
 *      → terrain_latent(32) + yaw(2), new_hidden
 *   3. Build proprio_full (53d): insert yaw at [6,7], zero foot_contacts
 * [49:53]
 *   4. Update obs_history with current proprio (yaw masked at [6:8] in history)
 *   5. Call history_encoder(obs_history) → hist_latent (20d)
 *   6. Build actor input: cat(proprio_53, terrain_latent_32, zeros_9,
 * hist_latent_20)
 *   7. Call actor_backbone(114d) → actions (12d)
 *   8. target = default_dof_pos + actions * action_scale
 */

#include "parkour_policy_runner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace deploy {
namespace {

template <size_t N>
torch::Tensor tensor_from_float_array(const std::array<float, N> &arr,
                                      torch::Device device) {
  return torch::from_blob(const_cast<float *>(arr.data()),
                          {1, static_cast<long>(N)}, torch::kFloat32)
      .clone()
      .to(device);
}

} // namespace

ParkourPolicyRunner::ParkourPolicyRunner(const RobotRuntimeConfig &config)
    : config_(config), device_(config.device) {
  std::cout << "[ParkourPolicyRunner] Loading heightmap_encoder: "
            << config_.heightmap_encoder_path << std::endl;
  std::cout << "[ParkourPolicyRunner] Loading history_encoder: "
            << config_.history_encoder_path << std::endl;
  std::cout << "[ParkourPolicyRunner] Loading actor_backbone: "
            << config_.actor_backbone_path << std::endl;

  heightmap_encoder_ =
      torch::jit::load(config_.heightmap_encoder_path, device_);
  history_encoder_ = torch::jit::load(config_.history_encoder_path, device_);
  actor_backbone_ = torch::jit::load(config_.actor_backbone_path, device_);

  heightmap_encoder_.eval();
  history_encoder_.eval();
  actor_backbone_.eval();

  // Initialize GRU hidden state (1, 1, hidden_size=512)
  gru_hidden_ = torch::zeros(
      {1, 1, 512},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  // Initialize obs history buffer (1, HISTORY_LEN, N_PROPRIO)
  obs_history_ = torch::zeros(
      {1, HISTORY_LEN, N_PROPRIO},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  // Initialize last actions
  last_actions_ = torch::zeros(
      {1, NUM_ACTIONS},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  // Default dof pos tensor
  default_dof_pos_ = tensor_from_float_array(config_.default_dof_pos, device_);

  validate_model_shapes();
}

void ParkourPolicyRunner::validate_model_shapes() {
  torch::NoGradGuard no_grad;

  // Test heightmap_encoder
  auto test_hm = torch::zeros(
      {1, NUM_HEIGHT_POINTS},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto test_prop = torch::zeros(
      {1, N_PROPRIO_STUDENT},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto test_hidden = torch::zeros(
      {1, 1, 512},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  auto hm_result =
      heightmap_encoder_.forward({test_hm, test_prop, test_hidden});

  // Result should be a tuple: (latent_yaw, new_hidden)
  auto hm_tuple = hm_result.toTuple();
  auto latent_yaw = hm_tuple->elements()[0].toTensor();
  auto new_hidden = hm_tuple->elements()[1].toTensor();

  if (latent_yaw.size(1) != HEIGHTMAP_ENCODER_OUTPUT_DIM) {
    throw std::runtime_error(
        "[ParkourPolicyRunner] heightmap_encoder output dim mismatch: got " +
        std::to_string(latent_yaw.size(1)) + " expected " +
        std::to_string(HEIGHTMAP_ENCODER_OUTPUT_DIM));
  }

  // Test history_encoder
  auto test_hist = torch::zeros(
      {1, HISTORY_LEN, N_PROPRIO},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  // history_encoder expects (nd, T, n_proprio) reshaped to (nd*T, n_proprio)
  // but since we export with reshape inside, we pass (1, 10, 53) directly
  auto hist_result = history_encoder_.forward({test_hist}).toTensor();
  if (hist_result.size(1) != PRIV_ENCODER_OUTPUT_DIM) {
    throw std::runtime_error(
        "[ParkourPolicyRunner] history_encoder output dim mismatch: got " +
        std::to_string(hist_result.size(1)) + " expected " +
        std::to_string(PRIV_ENCODER_OUTPUT_DIM));
  }

  // Test actor_backbone
  auto test_input = torch::zeros(
      {1, ACTOR_INPUT_DIM},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto act_result = actor_backbone_.forward({test_input}).toTensor();
  if (act_result.size(1) != NUM_ACTIONS) {
    throw std::runtime_error(
        "[ParkourPolicyRunner] actor_backbone output dim mismatch: got " +
        std::to_string(act_result.size(1)) + " expected " +
        std::to_string(NUM_ACTIONS));
  }

  std::cout << "[ParkourPolicyRunner] Shape check OK: "
            << "heightmap_encoder (132+49+hidden) -> "
            << HEIGHTMAP_ENCODER_OUTPUT_DIM << ", history_encoder (10x53) -> "
            << PRIV_ENCODER_OUTPUT_DIM << ", actor_backbone ("
            << ACTOR_INPUT_DIM << ") -> " << NUM_ACTIONS << std::endl;
}

void ParkourPolicyRunner::reset() {
  gru_hidden_.zero_();
  obs_history_.zero_();
  last_actions_.zero_();
  infer_count_ = 0;
}

torch::Tensor ParkourPolicyRunner::build_proprio_student(
    const std::array<float, 3> &commands, const std::array<float, 3> &ang_vel,
    const std::array<float, 3> &projected_gravity,
    const std::array<float, NUM_JOINTS> &dof_pos,
    const std::array<float, NUM_JOINTS> &dof_vel) {
  // Build 49-dim student proprio (no foot contacts)
  // Layout matches compute_observations in training:
  //   [0:3]   base_ang_vel * ang_vel_scale
  //   [3:5]   roll, pitch (from projected_gravity → euler approx)
  //   [5]     0 (masked delta_yaw placeholder)
  //   [6:8]   0, 0 (yaw placeholders, will be replaced by encoder estimate)
  //   [8:10]  0, 0 (masked commands)
  //   [10]    cmd_vx * command_scale
  //   [11]    1.0 (env_class != 17, non-gap terrain)
  //   [12]    0.0 (env_class == 17, gap terrain)
  //   [13:25] (dof_pos - default) * dof_pos_scale
  //   [25:37] dof_vel * dof_vel_scale
  //   [37:49] last_actions

  auto proprio = torch::zeros(
      {1, N_PROPRIO_STUDENT},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto ptr = proprio.data_ptr<float>();

  // [0:3] base_ang_vel * scale
  ptr[0] = ang_vel[0] * config_.ang_vel_scale;
  ptr[1] = ang_vel[1] * config_.ang_vel_scale;
  ptr[2] = ang_vel[2] * config_.ang_vel_scale;

  // [3:5] roll, pitch from projected gravity
  // In training: roll = atan2(gx_component, gz_component), pitch = asin(...)
  // But the training code uses: imu_obs = torch.stack((self.roll, self.pitch))
  // which are euler angles from quaternion. The projected_gravity gives us the
  // gravity vector in body frame. We can compute roll/pitch from it:
  // roll  = atan2(gy, gz) -> but training uses atan2(2*(w*x+y*z), 1-2*(x²+y²))
  // For deployment, projected_gravity = [gx, gy, gz] where gz ≈ -1 on flat
  // ground roll  ≈ atan2(-gy, -gz) pitch ≈ asin(gx)
  const float gx = projected_gravity[0];
  const float gy = projected_gravity[1];
  const float gz = projected_gravity[2];
  ptr[3] = std::atan2(-gy, -gz);                   // roll
  ptr[4] = std::asin(std::clamp(gx, -1.0f, 1.0f)); // pitch

  // [5] 0 (masked)
  ptr[5] = 0.0f;

  // [6:8] yaw placeholders (will be replaced after heightmap_encoder)
  ptr[6] = 0.0f;
  ptr[7] = 0.0f;

  // [8:10] masked commands
  ptr[8] = 0.0f;
  ptr[9] = 0.0f;

  // [10] cmd_vx
  ptr[10] = commands[0] * config_.command_scale;

  // [11:13] terrain type flags (default: non-gap)
  ptr[11] = 1.0f; // (env_class != 17)
  ptr[12] = 0.0f; // (env_class == 17)

  // [13:25] (dof_pos - default) * dof_pos_scale
  for (int i = 0; i < NUM_JOINTS; ++i) {
    ptr[13 + i] =
        (dof_pos[i] - config_.default_dof_pos[i]) * config_.dof_pos_scale;
  }

  // [25:37] dof_vel * dof_vel_scale
  for (int i = 0; i < NUM_JOINTS; ++i) {
    ptr[25 + i] = dof_vel[i] * config_.dof_vel_scale;
  }

  // [37:49] last_actions
  auto la_ptr = last_actions_.cpu().data_ptr<float>();
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    ptr[37 + i] = la_ptr[i];
  }

  return proprio;
}

torch::Tensor
ParkourPolicyRunner::build_proprio_full(const torch::Tensor &proprio_student,
                                        const torch::Tensor &yaw_estimate) {
  // Build 53-dim full proprio by appending zeroed foot_contacts
  auto proprio_full = torch::zeros(
      {1, N_PROPRIO},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  // Copy student proprio (49d)
  proprio_full.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(0, N_PROPRIO_STUDENT)},
      proprio_student);

  // Insert yaw estimates at [6:8] (scaled by yaw_scale)
  proprio_full.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(6, 8)},
      yaw_estimate * config_.yaw_scale);

  // [49:53] foot_contacts remain zero

  return proprio_full;
}

void ParkourPolicyRunner::update_obs_history(
    const torch::Tensor &proprio_full) {
  // Mask yaw in history (set [6:8] to 0 for history storage)
  auto obs_for_history = proprio_full.clone();
  obs_for_history.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(6, 8)},
      torch::zeros(
          {1, 2},
          torch::TensorOptions().dtype(torch::kFloat32).device(device_)));

  // Shift history and append new observation
  // obs_history_: (1, 10, 53) → shift left, append at end
  obs_history_ = torch::cat(
      {obs_history_.index({torch::indexing::Slice(),
                           torch::indexing::Slice(1, torch::indexing::None),
                           torch::indexing::Slice()}),
       obs_for_history.unsqueeze(1)},
      1);
}

std::array<float, NUM_JOINTS> ParkourPolicyRunner::target_from_actions(
    const std::array<float, NUM_ACTIONS> &actions) const {
  std::array<float, NUM_JOINTS> target{};
  for (int i = 0; i < NUM_JOINTS; ++i) {
    target[i] = config_.default_dof_pos[i] + actions[i] * config_.action_scale;
    target[i] = std::clamp(target[i], config_.joint_pos_lower[i],
                           config_.joint_pos_upper[i]);
  }
  return target;
}

void ParkourPolicyRunner::step(
    const std::array<float, 3> &commands, const std::array<float, 3> &ang_vel,
    const std::array<float, 3> &projected_gravity,
    const std::array<float, NUM_JOINTS> &dof_pos,
    const std::array<float, NUM_JOINTS> &dof_vel,
    const std::array<float, NUM_HEIGHT_POINTS> &height_measurements,
    std::array<float, NUM_JOINTS> &target_dof_pos,
    std::array<float, NUM_ACTIONS> &actions) {
  torch::NoGradGuard no_grad;

  // 1. Build student proprioception (49d)
  auto proprio_student = build_proprio_student(
      commands, ang_vel, projected_gravity, dof_pos, dof_vel);

  // 2. Prepare heightmap tensor
  // Height measurements are already pre-processed by height_subscriber
  // Here we apply the same transform as training: clip(distance - 0.3, -1, 1)
  // The height_subscriber provides raw distances; we pass them directly
  auto height_tensor = tensor_from_float_array(height_measurements, device_);
  height_tensor =
      torch::clamp(height_tensor - config_.height_bias, -1.0f, 1.0f);

  // Mask yaw in student proprio for encoder input (set [6:8] to 0)
  auto proprio_for_encoder = proprio_student.clone();
  proprio_for_encoder.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(6, 8)},
      torch::zeros(
          {1, 2},
          torch::TensorOptions().dtype(torch::kFloat32).device(device_)));

  // 3. Run heightmap_encoder: (heightmap, proprio, hidden) → (latent_yaw,
  // new_hidden)
  auto hm_result = heightmap_encoder_.forward(
      {height_tensor, proprio_for_encoder, gru_hidden_});
  auto hm_tuple = hm_result.toTuple();
  auto latent_yaw = hm_tuple->elements()[0].toTensor(); // (1, 34)
  gru_hidden_ = hm_tuple->elements()[1].toTensor();     // (1, 1, 512)

  // Split terrain latent and yaw estimate
  auto terrain_latent = latent_yaw.index(
      {torch::indexing::Slice(),
       torch::indexing::Slice(0, HEIGHTMAP_LATENT_DIM)}); // (1, 32)
  auto yaw_estimate = latent_yaw.index(
      {torch::indexing::Slice(),
       torch::indexing::Slice(HEIGHTMAP_LATENT_DIM,
                              torch::indexing::None)}); // (1, 2)

  // 4. Build full proprio (53d) with yaw inserted and foot_contacts zeroed
  auto proprio_full = build_proprio_full(proprio_student, yaw_estimate);

  // 5. Update observation history
  update_obs_history(proprio_full);

  // 6. Run history_encoder: (1, 10, 53) → (1, 20)
  auto hist_latent = history_encoder_.forward({obs_history_}).toTensor();

  // 7. Build actor backbone input: cat(proprio_53, terrain_latent_32,
  // priv_explicit_9, hist_latent_20) = 114
  auto priv_explicit = torch::zeros(
      {1, N_PRIV_EXPLICIT},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  auto actor_input =
      torch::cat({proprio_full, terrain_latent, priv_explicit, hist_latent}, 1);

  // 8. Run actor_backbone: (1, 114) → (1, 12)
  auto action_tensor = actor_backbone_.forward({actor_input}).toTensor();

  // Clip actions
  action_tensor =
      torch::clamp(action_tensor, -config_.clip_actions, config_.clip_actions);

  // Store last actions
  last_actions_ = action_tensor.clone();

  // 9. Extract actions and compute targets
  auto action_cpu = action_tensor.cpu().contiguous();
  const float *act_ptr = action_cpu.data_ptr<float>();
  std::copy(act_ptr, act_ptr + NUM_ACTIONS, actions.begin());
  target_dof_pos = target_from_actions(actions);

  ++infer_count_;
  if (config_.debug_print_policy &&
      infer_count_ % static_cast<uint64_t>(config_.debug_print_interval) ==
          0U) {
    auto yaw_cpu = yaw_estimate.cpu().contiguous();
    std::cout << "\n[ParkourPolicyRunner] step=" << infer_count_ << " cmd=["
              << commands[0] << "," << commands[1] << "," << commands[2]
              << "] action[0:3]=[" << actions[0] << "," << actions[1] << ","
              << actions[2] << "] yaw_est=[" << yaw_cpu.data_ptr<float>()[0]
              << "," << yaw_cpu.data_ptr<float>()[1] << "]" << std::endl;
  }
}

} // namespace deploy
