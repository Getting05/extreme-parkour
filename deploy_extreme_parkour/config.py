from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class RobotConfig:
    name: str = "mybot_v3"
    joint_names: List[str] = field(default_factory=lambda: [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ])
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
        "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
    })


@dataclass
class ControlConfig:
    dt: float = 0.02
    action_scale: float = 0.25
    clip_actions: float = 1.2


@dataclass
class ObsConfig:
    n_proprio_full: int = 53
    n_proprio_student: int = 49
    n_scan: int = 132
    n_priv_explicit: int = 9
    n_priv_latent: int = 29
    history_len: int = 10
    n_heightmap: int = 132
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    height_scale: float = 5.0
    yaw_scale: float = 1.5


@dataclass
class HeightmapConfig:
    clip_min: float = -1.0
    clip_max: float = 1.0
    base_height_offset: float = 0.3
    measured_points_x: List[float] = field(default_factory=lambda: [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2])
    measured_points_y: List[float] = field(default_factory=lambda: [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75])


@dataclass
class PolicyConfig:
    actor_path: str = "./traced/mybot_v3_heightmap_actor.pt"
    encoder_path: str = "./traced/mybot_v3_heightmap_encoder.pt"
    device: str = "cpu"


@dataclass
class MujocoConfig:
    model_path: str = "./resources/robots/mybot_v3/mjcf/mybot_v3.xml"
    render: bool = True
    duration_s: float = 60.0


@dataclass
class DeployConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    heightmap: HeightmapConfig = field(default_factory=HeightmapConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)


def _update_dataclass(obj, values: dict):
    for key, value in values.items():
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None) -> DeployConfig:
    cfg = DeployConfig()
    if path is None:
        return cfg
    data = yaml.safe_load(Path(path).read_text()) or {}
    _update_dataclass(cfg, data)
    return cfg
