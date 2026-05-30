from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .observation import SensorFrame
from .policy import HeightmapStudentPolicy


def quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)


def projected_gravity_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    rot = quat_to_rotmat_xyzw(q)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return rot.T @ g_world


def sample_flat_heightmap(cfg) -> np.ndarray:
    return np.zeros(cfg.obs.n_heightmap, dtype=np.float32)


def run(args):
    import mujoco
    import mujoco.viewer

    cfg = load_config(args.config)
    if args.model is not None:
        cfg.mujoco.model_path = args.model
    if args.actor is not None:
        cfg.policy.actor_path = args.actor
    if args.encoder is not None:
        cfg.policy.encoder_path = args.encoder
    cfg.mujoco.render = not args.headless

    model_path = Path(cfg.mujoco.model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    policy = HeightmapStudentPolicy(cfg)

    joint_ids = []
    actuator_ids = []
    for name in cfg.robot.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint not found in MuJoCo model: {name}")
        joint_ids.append(jid)
        aname = name.replace("_joint", "")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        actuator_ids.append(aid if aid >= 0 else len(actuator_ids))

    qadr = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32)
    dadr = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=np.int32)
    default_q = np.array([cfg.robot.default_joint_angles[n] for n in cfg.robot.joint_names], dtype=np.float32)
    data.qpos[qadr] = default_q
    mujoco.mj_forward(model, data)

    last_action = np.zeros(len(cfg.robot.joint_names), dtype=np.float32)
    start_time = time.time()

    def step_once():
        nonlocal last_action
        q = data.qpos[qadr].copy().astype(np.float32)
        dq = data.qvel[dadr].copy().astype(np.float32)
        quat_wxyz = data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_g = projected_gravity_from_quat_xyzw(quat_xyzw)
        base_ang_vel = data.qvel[3:6].copy().astype(np.float32)

        frame = SensorFrame(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_g,
            dof_pos=q,
            dof_vel=dq,
            last_action=last_action,
            goal_rel_body=np.array([args.goal_x, args.goal_y], dtype=np.float32),
            next_goal_rel_body=np.array([args.next_goal_x, args.next_goal_y], dtype=np.float32),
            target_yaw=float(args.target_yaw),
            heightmap=sample_flat_heightmap(cfg),
        )
        action, debug = policy.act(frame)
        last_action = action.copy()
        target_q = policy.action_to_target_q(action)
        data.ctrl[actuator_ids] = target_q
        mujoco.mj_step(model, data)
        return debug

    if cfg.mujoco.render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and time.time() - start_time < cfg.mujoco.duration_s:
                debug = step_once()
                viewer.sync()
                if args.print_debug:
                    print(debug)
    else:
        steps = int(cfg.mujoco.duration_s / cfg.control.dt)
        for i in range(steps):
            debug = step_once()
            if args.print_debug and i % 50 == 0:
                print(i, debug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--actor", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--print_debug", action="store_true")
    parser.add_argument("--goal_x", type=float, default=1.0)
    parser.add_argument("--goal_y", type=float, default=0.0)
    parser.add_argument("--next_goal_x", type=float, default=1.5)
    parser.add_argument("--next_goal_y", type=float, default=0.0)
    parser.add_argument("--target_yaw", type=float, default=0.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
