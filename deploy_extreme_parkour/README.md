# mybot_v3 Extreme Parkour heightmap deploy package

This package follows the same split used in the elmap deployment workflow: a shared policy runtime, an observation builder, a simulator runner, and a replay utility for checking recorded sensor streams before any hardware integration.

## Files

- `config.py`: dataclass-based deployment configuration.
- `observation.py`: converts runtime sensor values into the 49-value student proprioception, 53-value actor proprioception, 132-value heightmap, and 10-frame history layout.
- `policy.py`: loads the exported heightmap encoder and actor TorchScript modules, injects terrain latent into actor inference, and converts policy actions to target joint positions.
- `sim2sim_mujoco.py`: MuJoCo runner for IsaacGym-to-MuJoCo sim2sim checks.
- `replay_csv.py`: CSV replay utility for checking observation and action consistency from logged data.

## Export policy

From `legged_gym/legged_gym/scripts`:

```bash
python save_heightmap_jit.py --exptid 002-01-heightmap-student --proj_name rough_mybot_v3 --checkpoint -1 --device cpu
```

It exports:

- `<run>/traced/<exptid>-<checkpoint>-heightmap_actor.pt`
- `<run>/traced/<exptid>-<checkpoint>-heightmap_encoder.pt`

## Run MuJoCo sim2sim

```bash
python -m deploy_extreme_parkour.sim2sim_mujoco \
  --model resources/robots/mybot_v3/mjcf/mybot_v3.xml \
  --actor legged_gym/logs/rough_mybot_v3/002-01-heightmap-student/traced/002-01-heightmap-student-final-heightmap_actor.pt \
  --encoder legged_gym/logs/rough_mybot_v3/002-01-heightmap-student/traced/002-01-heightmap-student-final-heightmap_encoder.pt
```

Use `--headless --print_debug` for batch checks.

## Replay logged observations

The replay CSV must contain these columns:

- `base_ang_vel`: three numbers
- `projected_gravity`: three numbers
- `dof_pos`: twelve numbers
- `dof_vel`: twelve numbers
- `heightmap`: one hundred thirty-two numbers

Optional columns are `goal_rel_body`, `next_goal_rel_body`, `target_yaw`, and `t`.

```bash
python -m deploy_extreme_parkour.replay_csv --csv logs/sample.csv --out logs/actions.csv --actor <actor.pt> --encoder <encoder.pt>
```

## Current limitations

The MuJoCo runner uses a flat zero heightmap unless you replace `sample_flat_heightmap` with your own map sampler. That is intentional: first verify policy loading, joint order, observation shape, default pose, action scale, and inference stability, then connect the heightmap sampler.
