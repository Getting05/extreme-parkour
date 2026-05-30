/**
 * @file robot_config.h
 * @brief Compile-time dimensions for extreme-parkour heightmap student deployment.
 *
 * Observation layout (student, no foot contacts, no depth camera):
 *   proprio_student: ang_vel(3) + imu(2) + yaw_masked(1) + yaw(2) + cmd_masked(2)
 *                    + cmd_vx(1) + terrain_flags(2) + dof_pos(12) + dof_vel(12) + actions(12) = 49
 *   proprio_full:    proprio_student + foot_contacts(4) = 53  (foot_contacts zeroed)
 *   scan:            132 (teacher scan, replaced by heightmap latent in student)
 *   priv_explicit:   9  (zeroed in student)
 *   priv_latent:     29 (encoded by history_encoder from obs history)
 *   history:         10 * 53 = 530
 *   Total obs:       53 + 132 + 9 + 29 + 530 = 753
 */
#pragma once

namespace deploy {

constexpr int NUM_JOINTS = 12;
constexpr int NUM_ACTIONS = 12;

// Height measurement grid from terrain config
// measured_points_x: 12 points, measured_points_y: 11 points
constexpr int NUM_HEIGHT_POINTS_X = 12;
constexpr int NUM_HEIGHT_POINTS_Y = 11;
constexpr int NUM_HEIGHT_POINTS = NUM_HEIGHT_POINTS_X * NUM_HEIGHT_POINTS_Y;  // 132

// Proprioception dimensions
constexpr int N_PROPRIO_STUDENT = 49;   // Without foot contacts
constexpr int N_PROPRIO = 53;           // Full teacher proprio (foot_contacts zeroed)
constexpr int N_FOOT_CONTACTS = 4;     // foot_contacts dimension

// Scan / terrain encoder dimensions
constexpr int N_SCAN = 132;            // Teacher scan points (replaced by latent in student)
constexpr int SCAN_ENCODER_OUTPUT_DIM = 32;   // scan_encoder output
constexpr int HEIGHTMAP_LATENT_DIM = 32;      // heightmap encoder terrain latent
constexpr int HEIGHTMAP_YAW_DIM = 2;          // heightmap encoder yaw estimate
constexpr int HEIGHTMAP_ENCODER_OUTPUT_DIM = HEIGHTMAP_LATENT_DIM + HEIGHTMAP_YAW_DIM;  // 34

// Privileged information dimensions
constexpr int N_PRIV_EXPLICIT = 9;     // base_lin_vel * 3 (zeroed in student)
constexpr int N_PRIV_LATENT = 29;      // mass(4) + friction(1) + motor_strength(24)
constexpr int PRIV_ENCODER_OUTPUT_DIM = 20;  // priv_encoder output (from [64, 20])

// History encoding
constexpr int HISTORY_LEN = 10;

// Actor backbone input dimension
constexpr int ACTOR_INPUT_DIM = N_PROPRIO + SCAN_ENCODER_OUTPUT_DIM +
                                 N_PRIV_EXPLICIT + PRIV_ENCODER_OUTPUT_DIM;  // 53+32+9+20=114

// Full observation buffer size (for heightmap_actor)
constexpr int FULL_OBS_DIM = N_PROPRIO + N_SCAN + N_PRIV_EXPLICIT +
                              N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO;  // 753

} // namespace deploy
