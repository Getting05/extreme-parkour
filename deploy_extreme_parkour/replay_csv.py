from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import load_config
from .observation import SensorFrame
from .policy import HeightmapStudentPolicy


def vec(text: str, n: int) -> np.ndarray:
    xs = [float(x) for x in text.replace(',', ' ').split()]
    if len(xs) != n:
        raise ValueError(f'expected {n} values, got {len(xs)}')
    return np.asarray(xs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--actor', type=str, default=None)
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out', type=str, default='heightmap_policy_replay.csv')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.actor:
        cfg.policy.actor_path = args.actor
    if args.encoder:
        cfg.policy.encoder_path = args.encoder

    policy = HeightmapStudentPolicy(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline='') as fin, open(out_path, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        fields = ['t'] + [f'a{i}' for i in range(len(cfg.robot.joint_names))] + ['obs_norm', 'action_norm']
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            action_prev = policy.last_action.copy()
            frame = SensorFrame(
                base_ang_vel=vec(row['base_ang_vel'], 3),
                projected_gravity=vec(row['projected_gravity'], 3),
                dof_pos=vec(row['dof_pos'], len(cfg.robot.joint_names)),
                dof_vel=vec(row['dof_vel'], len(cfg.robot.joint_names)),
                last_action=action_prev,
                goal_rel_body=vec(row.get('goal_rel_body', '1 0'), 2),
                next_goal_rel_body=vec(row.get('next_goal_rel_body', '1.5 0'), 2),
                target_yaw=float(row.get('target_yaw', 0.0)),
                heightmap=vec(row['heightmap'], cfg.obs.n_heightmap),
            )
            action, info = policy.act(frame)
            writer.writerow({
                't': row.get('t', ''),
                **{f'a{i}': float(action[i]) for i in range(action.shape[0])},
                'obs_norm': info['obs_norm'],
                'action_norm': info['action_norm'],
            })
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
