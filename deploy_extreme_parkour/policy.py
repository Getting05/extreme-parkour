from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .config import DeployConfig
from .observation import ObservationBuilder, SensorFrame


class HeightmapStudentPolicy:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.policy.device)
        self.actor = torch.jit.load(str(Path(cfg.policy.actor_path)), map_location=self.device)
        self.encoder = torch.jit.load(str(Path(cfg.policy.encoder_path)), map_location=self.device)
        self.actor.eval()
        self.encoder.eval()
        self.obs_builder = ObservationBuilder(cfg)
        self.last_action = np.zeros(len(cfg.robot.joint_names), dtype=np.float32)

    def reset(self):
        self.obs_builder.reset()
        self.last_action[:] = 0.0
        if hasattr(self.encoder, "reset_hidden_states"):
            self.encoder.reset_hidden_states()

    @torch.no_grad()
    def act(self, frame: SensorFrame) -> Tuple[np.ndarray, dict]:
        full0 = self.obs_builder.build_proprio_full(frame)
        student_prop = self.obs_builder.build_student_proprio(full0)
        heightmap = self.obs_builder.build_heightmap(frame)

        h_t = torch.from_numpy(heightmap).float().unsqueeze(0).to(self.device)
        p_t = torch.from_numpy(student_prop).float().unsqueeze(0).to(self.device)
        latent_yaw = self.encoder(h_t, p_t)
        terrain_latent = latent_yaw[:, :-2]
        yaw = latent_yaw[:, -2:]

        full = self.obs_builder.build_proprio_full(frame, yaw_estimate=yaw.cpu().numpy()[0])
        obs = self.obs_builder.build_actor_obs(full)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor(obs_t, terrain_latent)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        action_np = np.clip(action_np, -self.cfg.control.clip_actions, self.cfg.control.clip_actions)
        self.last_action = action_np.copy()
        debug = {
            "terrain_latent": terrain_latent.squeeze(0).cpu().numpy(),
            "yaw": yaw.squeeze(0).cpu().numpy(),
            "obs_norm": float(np.linalg.norm(obs)),
            "action_norm": float(np.linalg.norm(action_np)),
        }
        return action_np, debug

    def action_to_target_q(self, action: np.ndarray, current_q: np.ndarray | None = None) -> np.ndarray:
        default_q = np.array([self.cfg.robot.default_joint_angles[n] for n in self.cfg.robot.joint_names], dtype=np.float32)
        return default_q + self.cfg.control.action_scale * np.asarray(action, dtype=np.float32)
