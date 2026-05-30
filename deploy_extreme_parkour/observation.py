from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import DeployConfig


@dataclass
class SensorFrame:
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    last_action: np.ndarray
    goal_rel_body: np.ndarray
    next_goal_rel_body: np.ndarray
    target_yaw: float
    heightmap: np.ndarray


class ObservationBuilder:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        n_full = cfg.obs.n_proprio_full
        self.history = np.zeros((cfg.obs.history_len, n_full), dtype=np.float32)
        self.default_dof_pos = np.array(
            [cfg.robot.default_joint_angles[name] for name in cfg.robot.joint_names],
            dtype=np.float32,
        )

    def reset(self):
        self.history[:] = 0.0

    def build_proprio_full(self, frame: SensorFrame, yaw_estimate: Optional[np.ndarray] = None) -> np.ndarray:
        cfg = self.cfg
        obs = np.zeros(cfg.obs.n_proprio_full, dtype=np.float32)

        base_ang_vel = np.asarray(frame.base_ang_vel, dtype=np.float32) * cfg.obs.ang_vel_scale
        projected_gravity = np.asarray(frame.projected_gravity, dtype=np.float32)
        goal = np.asarray(frame.goal_rel_body, dtype=np.float32)
        next_goal = np.asarray(frame.next_goal_rel_body, dtype=np.float32)
        dof_pos = (np.asarray(frame.dof_pos, dtype=np.float32) - self.default_dof_pos) * cfg.obs.dof_pos_scale
        dof_vel = np.asarray(frame.dof_vel, dtype=np.float32) * cfg.obs.dof_vel_scale
        last_action = np.asarray(frame.last_action, dtype=np.float32)

        obs[0:3] = base_ang_vel
        obs[3:5] = projected_gravity[:2]
        obs[5:8] = np.array([goal[0], goal[1], frame.target_yaw], dtype=np.float32)
        if yaw_estimate is not None:
            obs[6:8] = cfg.obs.yaw_scale * np.asarray(yaw_estimate, dtype=np.float32)[-2:]
        obs[8:12] = np.array([next_goal[0], next_goal[1], 0.0, 0.0], dtype=np.float32)
        obs[12:24] = dof_pos
        obs[24:36] = dof_vel
        obs[36:48] = last_action
        obs[48:52] = 0.0
        obs[52] = 0.0
        return obs

    def build_student_proprio(self, full_proprio: np.ndarray) -> np.ndarray:
        full = np.asarray(full_proprio, dtype=np.float32)
        return np.concatenate([full[:48], full[52:53]], axis=0).astype(np.float32)

    def build_heightmap(self, frame: SensorFrame) -> np.ndarray:
        h = np.asarray(frame.heightmap, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.cfg.obs.n_heightmap:
            raise ValueError(f"heightmap length {h.shape[0]} != {self.cfg.obs.n_heightmap}")
        h = np.clip(h, self.cfg.heightmap.clip_min, self.cfg.heightmap.clip_max)
        return h.astype(np.float32)

    def update_history(self, full_proprio: np.ndarray) -> np.ndarray:
        self.history[:-1] = self.history[1:]
        self.history[-1] = np.asarray(full_proprio, dtype=np.float32)
        return self.history.reshape(-1).astype(np.float32)

    def build_actor_obs(self, full_proprio: np.ndarray) -> np.ndarray:
        history_flat = self.update_history(full_proprio)
        zeros_scan = np.zeros(self.cfg.obs.n_scan, dtype=np.float32)
        zeros_priv = np.zeros(self.cfg.obs.n_priv_explicit + self.cfg.obs.n_priv_latent, dtype=np.float32)
        return np.concatenate([full_proprio, zeros_scan, zeros_priv, history_flat], axis=0).astype(np.float32)
