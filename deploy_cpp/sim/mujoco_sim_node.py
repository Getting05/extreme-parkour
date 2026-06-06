#!/usr/bin/env python3
"""
MuJoCo simulation node for Extreme-Parkour quadruped robot.

Bridges MuJoCo physics simulation with the C++ deploy_node via ROS2 topics:
    - Subscribes: /mujoco/joint_cmd
            - 12 floats: target positions
            - 14 floats: target + scalar kp/kd
            - 36 floats: target + per-joint kp[12] + per-joint kd[12]
  - Publishes:  /mujoco/joint_state (Float32MultiArray, 24 floats: 12 pos + 12 vel)
  - Publishes:  /fast_livo2/state6_imu_prop  (Float32MultiArray, 6 floats: 3 ang_vel + 3 proj_grav)
  - Publishes:  /height_measurements (Float32MultiArray, 132 processed heightmap obs values)
  - Publishes:  /parkour/goal_yaw (Float32MultiArray, [0, delta_yaw, delta_next_yaw])
  - Publishes:  /joint_states       (JointState, for RViz visualization)

Usage:
  conda activate mujoco_sim
  source /opt/ros/humble/setup.bash
  python3 sim/mujoco_sim_node.py

  Or via launch:
  ros2 launch deploy_cpp sim.launch.py
"""

import os
import sys
import time
import argparse
import threading
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import yaml

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState

# MuJoCo
import mujoco
import mujoco.viewer

# ============================================================
# Robot configuration (must match robot_config.h)
# ============================================================
NUM_JOINTS = 12

# Joint names in DOF order

def load_robot_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    required = [
        'dt', 'decimation', 'num_of_dofs', 'default_dof_pos', 'joint_names',
        'joint_controller_names', 'torque_limits', 'kp_joint', 'kd_joint',
        'joint_transmission_ratio', 'mujoco_xml_relpath'
    ]
    for key in required:
        if key not in cfg:
            raise RuntimeError(f'Missing required key in robot yaml: {key}')

    if int(cfg['num_of_dofs']) != NUM_JOINTS:
        raise RuntimeError('num_of_dofs must be 12 for this node')

    if len(cfg['joint_names']) != NUM_JOINTS:
        raise RuntimeError('joint_names length must be 12')
    if len(cfg['joint_controller_names']) != NUM_JOINTS:
        raise RuntimeError('joint_controller_names length must be 12')
    if len(cfg['default_dof_pos']) != NUM_JOINTS:
        raise RuntimeError('default_dof_pos length must be 12')
    if len(cfg['torque_limits']) != NUM_JOINTS:
        raise RuntimeError('torque_limits length must be 12')
    if len(cfg['joint_transmission_ratio']) != NUM_JOINTS:
        raise RuntimeError('joint_transmission_ratio length must be 12')

    return cfg


def parse_scalar_or_array(value, length: int, name: str) -> np.ndarray:
    """Parse YAML scalar or length-N sequence into float32 numpy array."""
    if isinstance(value, (int, float)):
        return np.full(length, float(value), dtype=np.float32)
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise RuntimeError(f'{name} length must be {length}')
        return np.array(value, dtype=np.float32)
    raise RuntimeError(f'{name} must be scalar or length-{length} array')


def resolve_path(pkg_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(pkg_dir, path_value)


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_wxyz_to_roll_pitch_yaw(q):
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def generate_parkour_course(cfg: dict, seed=None, difficulty: float = 0.5) -> dict:
    """Generate the reference sim2sim parkour box course."""
    rng = np.random.default_rng(seed)
    terrain_length = float(cfg.get('terrain_length', 18.0))
    num_goals = int(cfg.get('num_goals', 8))
    num_stones = max(1, num_goals - 2)
    if difficulty < 0.0:
        difficulty = float(rng.uniform(0.7, 1.0))

    x_range = [-0.1, 0.1 + 0.3 * difficulty]
    y_range = [0.2, 0.3 + 0.1 * difficulty]
    stone_len_range = [0.9 - 0.3 * difficulty, 1.0 - 0.2 * difficulty]
    incline_height = 0.25 * difficulty
    last_incline_height = incline_height + 0.1 - 0.1 * difficulty
    stone_width = 1.0
    platform_len = 2.5
    platform_height = 0.0
    last_stone_len = 1.6
    robot_origin_x = 1.0

    stone_len = float(rng.uniform(*stone_len_range))
    stone_len = 2.0 * round(stone_len / 2.0, 1)
    dis_x_min = stone_len + x_range[0]
    dis_x_max = stone_len + x_range[1]

    goals = np.zeros((num_stones + 2, 3), dtype=np.float32)
    geoms = [
        {
            'name': 'parkour_start_platform',
            'pos': [platform_len / 2.0 - robot_origin_x, 0.0, platform_height / 2.0],
            'size': [platform_len / 2.0, 2.0, 0.025],
            'rgba': '0.45 0.45 0.45 1',
        }
    ]
    goals[0] = [platform_len - stone_len / 2.0 - robot_origin_x, 0.0, platform_height]

    dis_x = platform_len - float(rng.uniform(dis_x_min, dis_x_max)) + stone_len / 2.0
    left_right_flag = int(rng.integers(0, 2))
    dis_z = 0.0
    last_center_x = dis_x
    last_len = stone_len
    for i in range(num_stones):
        dis_x += float(rng.uniform(dis_x_min, dis_x_max))
        pos_neg = 1.0 if left_right_flag == 1 else -1.0
        dis_y = pos_neg * float(rng.uniform(y_range[0], y_range[1]))
        if i == num_stones - 1:
            dis_x += last_stone_len / 4.0
            length = last_stone_len
            height = last_incline_height
        else:
            length = stone_len
            height = incline_height
        slope_angle = float(np.arctan2(2.0 * height, stone_width) * pos_neg)
        goals[i + 1] = [dis_x - robot_origin_x, dis_y, dis_z]
        geoms.append(
            {
                'name': f'parkour_stone_{i}',
                'pos': [dis_x - robot_origin_x, dis_y, dis_z],
                'size': [length / 2.0, stone_width / 2.0, 0.03],
                'euler': [slope_angle, 0.0, 0.0],
                'rgba': '0.35 0.35 0.35 1',
            }
        )
        last_center_x = dis_x
        last_len = length
        left_right_flag = 1 - left_right_flag

    final_dis_x = last_center_x + 2.0 * float(rng.uniform(dis_x_min, dis_x_max))
    final_platform_start = last_center_x + last_len / 2.0 + 0.05
    final_len = max(3.0, terrain_length - final_platform_start)
    geoms.append(
        {
            'name': 'parkour_final_platform',
            'pos': [final_platform_start + final_len / 2.0 - robot_origin_x, 0.0, platform_height / 2.0],
            'size': [final_len / 2.0, 2.0, 0.025],
            'rgba': '0.45 0.45 0.45 1',
        }
    )
    goals[-1] = [final_dis_x - robot_origin_x, 0.0, platform_height]
    return {'kind': 'parkour', 'geoms': geoms, 'goals': goals}


class MujocoSubTerrain:
    def __init__(self, width, length, vertical_scale, horizontal_scale):
        self.width = int(width)
        self.length = int(length)
        self.vertical_scale = float(vertical_scale)
        self.horizontal_scale = float(horizontal_scale)
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
        self.goals = np.zeros((0, 2), dtype=np.float32)


def apply_terrain_padding(terrain, pad_width, pad_height):
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    if pad_width <= 0:
        return
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def add_random_uniform_roughness(terrain, min_height, max_height, step=0.005, downsampled_scale=0.075):
    if max_height <= min_height:
        return
    heights = np.arange(min_height, max_height + step, step)
    if heights.size == 0:
        return
    down_rows = max(2, int(terrain.width * terrain.horizontal_scale / downsampled_scale))
    down_cols = max(2, int(terrain.length * terrain.horizontal_scale / downsampled_scale))
    sampled = np.random.choice(heights, size=(down_rows, down_cols))
    row_idx = np.minimum(
        (np.linspace(0.0, 1.0, terrain.width) * (down_rows - 1)).astype(np.int32),
        down_rows - 1,
    )
    col_idx = np.minimum(
        (np.linspace(0.0, 1.0, terrain.length) * (down_cols - 1)).astype(np.int32),
        down_cols - 1,
    )
    rough = sampled[row_idx[:, None], col_idx[None, :]]
    terrain.height_field_raw += np.round(rough / terrain.vertical_scale).astype(np.int16)


def add_parkour_roughness(terrain, cfg, difficulty=1.0):
    height_cfg = cfg.get('terrain_height', cfg.get('height', [0.02, 0.06]))
    max_height = (float(height_cfg[1]) - float(height_cfg[0])) * difficulty + float(height_cfg[0])
    height = float(np.random.uniform(float(height_cfg[0]), max_height))
    add_random_uniform_roughness(
        terrain,
        min_height=-height,
        max_height=height,
        step=0.005,
        downsampled_scale=float(cfg.get('downsampled_scale', 0.075)),
    )


def parkour_hurdle_terrain_hf(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    stone_len=0.3,
    x_range=None,
    y_range=None,
    half_valid_width=None,
    hurdle_height_range=None,
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    x_range = [1.5, 2.4] if x_range is None else x_range
    y_range = [-0.4, 0.4] if y_range is None else y_range
    half_valid_width = [0.4, 0.8] if half_valid_width is None else half_valid_width
    hurdle_height_range = [0.2, 0.3] if hurdle_height_range is None else hurdle_height_range
    goals = np.zeros((num_stones + 2, 2), dtype=np.float32)
    mid_y = terrain.length // 2
    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height
    stone_len = round(stone_len / terrain.horizontal_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            x0 = max(dis_x - stone_len // 2, 0)
            x1 = min(dis_x + stone_len // 2, terrain.width)
            y0 = max(mid_y + rand_y - half_valid_width, 0)
            y1 = min(mid_y + rand_y + half_valid_width, terrain.length)
            terrain.height_field_raw[x0:x1, :] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[x0:x1, :y0] = 0
            terrain.height_field_raw[x0:x1, y1:] = 0
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - round(0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]
    terrain.goals = goals * terrain.horizontal_scale
    apply_terrain_padding(terrain, pad_width, pad_height)


def convert_heightfield_to_trimesh_np(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        move_corners[1:num_rows, 1:num_cols] -= hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3
    return vertices, triangles


def write_obj_mesh(path, vertices, triangles):
    with open(path, 'w', encoding='utf-8') as f:
        for v in vertices:
            f.write(f'v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n')
        for tri in triangles:
            f.write(f'f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n')


def heightfield_to_box_geoms(height_field_raw, horizontal_scale, vertical_scale, origin_shift, stride=2, height_step=0.02):
    hf_m = height_field_raw.astype(np.float32) * float(vertical_scale)
    rows, cols = hf_m.shape
    stride = max(1, int(stride))
    height_step = max(1e-4, float(height_step))
    bottom_z = float(hf_m.min() - 0.05)
    geoms = []
    for r0 in range(0, rows - 1, stride):
        r1 = min(r0 + stride, rows - 1)
        x0 = r0 * horizontal_scale
        x1 = r1 * horizontal_scale
        segments = []
        c0 = 0
        while c0 < cols - 1:
            c1 = min(c0 + stride, cols - 1)
            top = float(np.max(hf_m[r0:r1 + 1, c0:c1 + 1]))
            top = round(top / height_step) * height_step
            start_c = c0
            c0 = c1
            while c0 < cols - 1:
                next_c = min(c0 + stride, cols - 1)
                next_top = float(np.max(hf_m[r0:r1 + 1, c0:next_c + 1]))
                next_top = round(next_top / height_step) * height_step
                if next_top != top:
                    break
                c0 = next_c
            segments.append((start_c, c0, top))

        for start_c, end_c, top in segments:
            y0 = start_c * horizontal_scale
            y1 = end_c * horizontal_scale
            thickness = max(top - bottom_z, 0.002)
            geoms.append(
                {
                    'name': f'terrain_box_{len(geoms)}',
                    'pos': [
                        (x0 + x1) * 0.5 - float(origin_shift[0]),
                        (y0 + y1) * 0.5 - float(origin_shift[1]),
                        bottom_z + thickness * 0.5,
                    ],
                    'size': [
                        max((x1 - x0) * 0.5, 0.001),
                        max((y1 - y0) * 0.5, 0.001),
                        thickness * 0.5,
                    ],
                }
            )
    return geoms


def generate_mujoco_terrain_mesh(
    cfg: dict,
    terrain_kind='parkour_hurdle',
    seed=None,
    difficulty=0.8,
    collision_mode='box',
    box_stride=2,
    box_height_step=0.02,
):
    old_state = None
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(int(seed))
    try:
        num_goals = int(cfg.get('num_goals', 8))
        horizontal_scale = float(cfg.get('terrain_horizontal_scale', 0.05))
        vertical_scale = float(cfg.get('terrain_vertical_scale', 0.005))
        terrain_length = float(cfg.get('terrain_length', 18.0))
        terrain_width = float(cfg.get('terrain_width', 4.0))
        if difficulty < 0.0:
            difficulty = float(np.random.uniform(0.7, 1.0))

        terrain = MujocoSubTerrain(
            width=int(terrain_length / horizontal_scale),
            length=int(terrain_width / horizontal_scale),
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
        )
        num_obstacles = max(1, num_goals - 2)
        y_range = cfg.get('terrain_y_range', [-0.4, 0.4])
        if terrain_kind not in ('parkour_hurdle', 'parkour_flat'):
            raise ValueError(f'Unsupported heightfield terrain kind {terrain_kind!r}')
        parkour_hurdle_terrain_hf(
            terrain,
            num_stones=num_obstacles,
            stone_len=0.1 + 0.3 * difficulty,
            hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.25 * difficulty],
            pad_height=0,
            x_range=[1.2, 2.2],
            y_range=y_range,
            half_valid_width=[0.4, 0.8] if terrain_kind == 'parkour_hurdle' else [0.45, 1.0],
            flat=(terrain_kind == 'parkour_flat'),
        )
        if bool(cfg.get('terrain_roughness_enable', False)):
            add_parkour_roughness(terrain, cfg)
        vertices, triangles = convert_heightfield_to_trimesh_np(
            terrain.height_field_raw,
            horizontal_scale,
            vertical_scale,
            slope_threshold=float(cfg.get('terrain_slope_threshold', 1.5)),
        )
        origin_shift = np.array([1.0, terrain_width / 2.0, 0.0], dtype=np.float32)
        vertices = vertices - origin_shift
        goals = np.zeros((num_goals, 3), dtype=np.float32)
        goals[:, :2] = terrain.goals[:num_goals] - origin_shift[:2]

        mesh_file = tempfile.NamedTemporaryFile(prefix=f'mujoco_{terrain_kind}_', suffix='.obj', dir='/tmp', delete=False)
        mesh_file.close()
        write_obj_mesh(mesh_file.name, vertices, triangles)
        collision_geoms = []
        if collision_mode == 'box':
            collision_geoms = heightfield_to_box_geoms(
                terrain.height_field_raw,
                horizontal_scale,
                vertical_scale,
                origin_shift,
                stride=box_stride,
                height_step=box_height_step,
            )
        return {
            'kind': terrain_kind,
            'mesh_file': mesh_file.name,
            'collision_mode': collision_mode,
            'collision_geoms': collision_geoms,
            'goals': goals,
            'min_z': float(vertices[:, 2].min()),
            'max_z': float(vertices[:, 2].max()),
            'horizontal_scale': horizontal_scale,
            'vertical_scale': vertical_scale,
            'difficulty': float(difficulty),
        }
    finally:
        if old_state is not None:
            np.random.set_state(old_state)


def build_mujoco_xml_with_terrain(xml_path: str, terrain, normalize_dynamics: bool = True) -> str:
    if terrain is None and not normalize_dynamics:
        return xml_path

    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find('compiler')
    if compiler is not None:
        meshdir = compiler.get('meshdir')
        if meshdir and not os.path.isabs(meshdir):
            compiler.set('meshdir', os.path.abspath(os.path.join(os.path.dirname(xml_path), meshdir)))

    if normalize_dynamics:
        for joint in root.findall('.//joint'):
            if joint.get('type') == 'free':
                continue
            joint.set('damping', '0')
            joint.set('frictionloss', '0')
            joint.set('armature', '0')
        for motor in root.findall('.//motor'):
            motor.set('gear', '1')
            motor.set('ctrlrange', '-33.5 33.5')
            motor.set('ctrllimited', 'true')

    if terrain is None:
        tmp = tempfile.NamedTemporaryFile(prefix='mujoco_runtime_', suffix='.xml', dir='/tmp', delete=False)
        tmp.close()
        tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
        return tmp.name

    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('MuJoCo XML has no worldbody')

    floor = worldbody.find("./geom[@name='floor']")
    if floor is not None:
        floor.set('rgba', '0.08 0.08 0.08 1')

    if 'mesh_file' in terrain:
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        if floor is not None:
            floor.set('pos', f"0 0 {terrain['min_z'] - 0.2:.6f}")
        ET.SubElement(asset, 'mesh', {'name': 'parkour_heightfield_mesh', 'file': terrain['mesh_file']})
        ET.SubElement(
            worldbody,
            'geom',
            {
                'name': f"{terrain['kind']}_trimesh",
                'type': 'mesh',
                'mesh': 'parkour_heightfield_mesh',
                'rgba': '0.36 0.36 0.36 1',
                'contype': '0' if terrain.get('collision_mode') == 'box' else '1',
                'conaffinity': '0' if terrain.get('collision_mode') == 'box' else '1',
                'condim': '3',
                'friction': '1.0 1.0 0.0',
                'group': '1' if terrain.get('collision_mode') == 'box' else '0',
            },
        )
        for geom_cfg in terrain.get('collision_geoms', []):
            ET.SubElement(
                worldbody,
                'geom',
                {
                    'name': geom_cfg['name'],
                    'type': 'box',
                    'pos': '{:.6f} {:.6f} {:.6f}'.format(*geom_cfg['pos']),
                    'size': '{:.6f} {:.6f} {:.6f}'.format(*geom_cfg['size']),
                    'rgba': '0.36 0.36 0.36 0.08',
                    'contype': '1',
                    'conaffinity': '1',
                    'condim': '3',
                    'friction': '1.0 0.3 0.3',
                    'group': '3',
                },
            )
        tmp = tempfile.NamedTemporaryFile(prefix='mujoco_runtime_', suffix='.xml', dir='/tmp', delete=False)
        tmp.close()
        tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
        return tmp.name

    for geom_cfg in terrain.get('geoms', []):
        attrs = {
            'name': geom_cfg['name'],
            'type': 'box',
            'pos': '{:.6f} {:.6f} {:.6f}'.format(*geom_cfg['pos']),
            'size': '{:.6f} {:.6f} {:.6f}'.format(*geom_cfg['size']),
            'rgba': geom_cfg.get('rgba', '0.35 0.35 0.35 1'),
            'contype': '1',
            'conaffinity': '1',
            'condim': '3',
            'friction': '1.0 0.3 0.3',
        }
        if 'euler' in geom_cfg:
            attrs['euler'] = '{:.6f} {:.6f} {:.6f}'.format(*geom_cfg['euler'])
        ET.SubElement(worldbody, 'geom', attrs)

    tmp = tempfile.NamedTemporaryFile(prefix='mujoco_runtime_', suffix='.xml', dir='/tmp', delete=False)
    tmp.close()
    tree.write(tmp.name, encoding='utf-8', xml_declaration=True)
    return tmp.name


class MujocoSimNode(Node):
    """ROS2 node that runs MuJoCo simulation and publishes sensor data."""

    def __init__(self, xml_path: str, cfg: dict):
        super().__init__('mujoco_sim_node')
        self.cfg = cfg
        self.control_dt = float(cfg['control_dt']) if 'control_dt' in cfg else float(cfg['dt']) * int(cfg['decimation'])
        self.sim_dt = float(cfg['dt'])
        self.decimation = int(cfg['decimation'])
        self.joint_names = list(cfg['joint_names'])
        self.joint_controller_names = list(cfg['joint_controller_names'])
        self.default_dof_pos = np.array(cfg['default_dof_pos'], dtype=np.float32)
        self.torque_limits = np.array(cfg['torque_limits'], dtype=np.float32)
        self.joint_transmission_ratio = np.array(cfg['joint_transmission_ratio'], dtype=np.float32)
        self.height_points_x = np.array(
            cfg.get('height_points_x',
                    [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]),
            dtype=np.float32)
        self.height_points_y = np.array(
            cfg.get('height_points_y',
                    [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75]),
            dtype=np.float32)
        n_hx = self.height_points_x.size
        n_hy = self.height_points_y.size
        self.get_logger().info(f"Height grid: {n_hx}x{n_hy} = {n_hx*n_hy} points")
        if n_hx * n_hy < 1:
            raise RuntimeError('height grid must have at least 1 point')
        self.height_points = self._init_height_points()
        self.height_clip_min = float(cfg.get('height_clip_min', -1.0))
        self.height_clip_max = float(cfg.get('height_clip_max', 1.0))
        self.height_bias = float(cfg.get('height_bias', 0.3))

        self.mujoco_terrain = str(cfg.get('mujoco_terrain', 'flat'))
        self.terrain = self._build_terrain()
        self.goals = self._init_goals()
        self.cur_goal_idx = 0
        self.reach_goal_timer = 0.0
        self.goal_yaw = np.zeros(3, dtype=np.float32)
        self.last_heightmap_values = None
        self.last_heightmap_world_points = None
        self.visualize_heightmap = bool(cfg.get('visualize_heightmap', False))
        self.visualize_goals = bool(cfg.get('visualize_goals', False))
        self.visualize_goal_dirs = bool(cfg.get('visualize_goal_dirs', False))
        self.heightmap_marker_size = float(cfg.get('heightmap_marker_size', 0.025))
        self.goal_marker_size = float(cfg.get('goal_marker_size', 0.08))
        self.goal_dir_marker_radius = float(cfg.get('goal_dir_marker_radius', 0.018))

        # Optional pure-simulation ping-pong mode:
        # publish state -> wait for /mujoco/joint_cmd -> step physics
        self.declare_parameter('pingpong_mode', False)
        self.pingpong_mode = bool(self.get_parameter('pingpong_mode').value)

        # ---- Load MuJoCo model ----
        self.mujoco_xml_path = build_mujoco_xml_with_terrain(
            xml_path,
            self.terrain,
            normalize_dynamics=bool(cfg.get('mujoco_normalize_dynamics', False)),
        )
        self.get_logger().info(f'Loading MuJoCo model: {self.mujoco_xml_path}')
        self.model = mujoco.MjModel.from_xml_path(self.mujoco_xml_path)
        self.data = mujoco.MjData(self.model)

        # Verify timestep
        assert abs(self.model.opt.timestep - self.sim_dt) < 1e-6, \
            f"XML timestep {self.model.opt.timestep} != expected {self.sim_dt}"

        # ---- Build joint index mapping ----
        # Map JOINT_NAMES to MuJoCo joint qpos/qvel indices
        self.joint_qpos_idx = []  # qpos index for each DOF
        self.joint_qvel_idx = []  # qvel index for each DOF
        self.actuator_idx = []    # actuator index for each DOF

        for i, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                self.get_logger().error(f'Joint "{name}" not found in model!')
                sys.exit(1)
            # For hinge joints: qpos has 1 element, qvel has 1 element
            self.joint_qpos_idx.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_idx.append(self.model.jnt_dofadr[jid])

        # Map actuators in the exact policy DOF order from YAML.
        for name in self.joint_controller_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                self.get_logger().error(f'Actuator "{name}" not found in model!')
                sys.exit(1)
            self.actuator_idx.append(aid)

        # Find IMU site
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.raycast_bodyexclude = self._find_floating_base_body()
        self.robot_geom_ids = np.nonzero(self.model.geom_bodyid != 0)[0].astype(np.int32)
        self.raycast_geomgroup = np.ones(6, dtype=np.uint8)
        self.raycast_geomgroup[5] = 0

        # ---- Set initial pose ----
        mujoco.mj_resetData(self.model, self.data)
        init_base_pos = np.array(cfg.get('mujoco_initial_base_pos', [0.0, 0.0, 0.45]), dtype=np.float64)
        if self.model.nq >= 7 and init_base_pos.size == 3:
            self.data.qpos[:3] = init_base_pos
            self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for i in range(NUM_JOINTS):
            self.data.qpos[self.joint_qpos_idx[i]] = self.default_dof_pos[i]
        mujoco.mj_forward(self.model, self.data)

        # ---- Target positions (updated by subscriber) ----
        self.target_pos = self.default_dof_pos.copy()
        self.target_kp = parse_scalar_or_array(cfg['kp_joint'], NUM_JOINTS, 'kp_joint')
        self.target_kd = parse_scalar_or_array(cfg['kd_joint'], NUM_JOINTS, 'kd_joint')
        self.cmd_lock = threading.Lock()
        self.cmd_event = threading.Event()

        # ---- ROS2 QoS Profiles ----
        # Low-latency QoS for real-time control/sensor data:
        #   BEST_EFFORT, depth=1, VOLATILE
        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Standard QoS for visualization (latency-insensitive)
        qos_viz = QoSProfile(depth=10)

        # ---- ROS2 Publishers ----
        self.pub_joint_state = self.create_publisher(
            Float32MultiArray, '/mujoco/joint_state', qos_fast)

        self.pub_imu = self.create_publisher(
            Float32MultiArray, '/fast_livo2/state6_imu_prop', qos_fast)

        self.pub_height = self.create_publisher(
            Float32MultiArray, cfg.get('height_topic', '/height_measurements'), qos_fast)

        self.pub_goal_yaw = self.create_publisher(
            Float32MultiArray, cfg.get('goal_yaw_topic', '/parkour/goal_yaw'), qos_fast)

        self.pub_rviz_joint = self.create_publisher(
            JointState, '/joint_states', qos_viz)

        # ---- ROS2 Subscriber ----
        self.sub_cmd = self.create_subscription(
            Float32MultiArray, '/mujoco/joint_cmd', self.cmd_callback, qos_fast)

        self.get_logger().info('MuJoCo sim node initialized.')
        self.get_logger().info(f'  Sim DT: {self.sim_dt}s, Decimation: {self.decimation}, Control DT: {self.control_dt}s')
        self.get_logger().info(
            f'  PD gains (DOF0): kp={self.target_kp[0]:.3f}, kd={self.target_kd[0]:.3f}')
        self.get_logger().info(
            f'  Height topic: {cfg.get("height_topic", "/height_measurements")} ({self.height_points_x.size}x{self.height_points_y.size})')
        self.get_logger().info(
            f'  Goal yaw topic: {cfg.get("goal_yaw_topic", "/parkour/goal_yaw")} goals={len(self.goals)} terrain={self.mujoco_terrain}')

    def cmd_callback(self, msg: Float32MultiArray):
        """Receive target joint positions from deploy_node."""
        if len(msg.data) >= NUM_JOINTS:
            with self.cmd_lock:
                self.target_pos[:] = msg.data[:NUM_JOINTS]
                # Optional kp/kd payloads:
                # 14 = target + scalar kp/kd, 36 = target + kp[12] + kd[12]
                if len(msg.data) >= NUM_JOINTS * 3:
                    self.target_kp[:] = msg.data[NUM_JOINTS:2 * NUM_JOINTS]
                    self.target_kd[:] = msg.data[2 * NUM_JOINTS:3 * NUM_JOINTS]
                elif len(msg.data) >= NUM_JOINTS + 2:
                    self.target_kp.fill(msg.data[NUM_JOINTS])
                    self.target_kd.fill(msg.data[NUM_JOINTS + 1])
            self.cmd_event.set()

    def _build_terrain(self):
        if self.mujoco_terrain == 'flat':
            return None
        seed = int(self.cfg.get('terrain_seed', 1))
        difficulty = float(self.cfg.get('terrain_difficulty', 0.8))
        if self.mujoco_terrain in ('parkour_hurdle', 'parkour_flat'):
            collision = str(self.cfg.get('terrain_collision', 'box'))
            if collision not in ('box', 'mesh'):
                self.get_logger().warn(
                    f'Unsupported terrain_collision={collision!r}; falling back to box.')
                collision = 'box'
            self.get_logger().info(
                f'Generating {self.mujoco_terrain} terrain difficulty={difficulty} '
                f'seed={seed} collision={collision}')
            return generate_mujoco_terrain_mesh(
                self.cfg,
                terrain_kind=self.mujoco_terrain,
                seed=seed,
                difficulty=difficulty,
                collision_mode=collision,
                box_stride=int(self.cfg.get('terrain_box_stride', 2)),
                box_height_step=float(self.cfg.get('terrain_box_height_step', 0.02)),
            )
        if self.mujoco_terrain != 'parkour':
            self.get_logger().warn(
                f'Unsupported mujoco_terrain={self.mujoco_terrain!r}; falling back to parkour boxes.')
        self.get_logger().info(
            f'Generating parkour terrain difficulty={difficulty} seed={seed}')
        return generate_parkour_course(self.cfg, seed=seed, difficulty=difficulty)

    def _init_goals(self) -> np.ndarray:
        if self.terrain is not None and 'goals' in self.terrain:
            return np.asarray(self.terrain['goals'], dtype=np.float32)

        num_goals = int(self.cfg.get('num_goals', 8))
        terrain_length = float(self.cfg.get('terrain_length', 18.0))
        x_goals = np.linspace(1.0, max(1.0, terrain_length - 1.0), num_goals, dtype=np.float32)
        goals = np.zeros((num_goals, 3), dtype=np.float32)
        goals[:, 0] = x_goals
        return goals

    def _init_height_points(self) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(self.height_points_x, self.height_points_y, indexing='ij')
        points = np.zeros((grid_x.size, 3), dtype=np.float32)
        points[:, 0] = grid_x.reshape(-1)
        points[:, 1] = grid_y.reshape(-1)
        return points

    def _find_floating_base_body(self) -> int:
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_bodyid[jid])
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'body')
        return int(body_id) if body_id >= 0 else -1

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions [rad]."""
        return np.array([self.data.qpos[idx] for idx in self.joint_qpos_idx],
                        dtype=np.float32)

    def get_joint_vel(self) -> np.ndarray:
        """Get current joint velocities [rad/s]."""
        return np.array([self.data.qvel[idx] for idx in self.joint_qvel_idx],
                        dtype=np.float32)

    def get_imu_data(self) -> tuple:
        """
        Get IMU sensor data from MuJoCo.
        Returns: (ang_vel[3], projected_gravity[3])
        """
        # Angular velocity from gyro sensor
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "Body_Gyro")
        gyro_adr = self.model.sensor_adr[gyro_id]
        ang_vel = self.data.sensordata[gyro_adr:gyro_adr + 3].copy()

        # Compute projected gravity from body quaternion
        quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "Body_Quat")
        quat_adr = self.model.sensor_adr[quat_id]
        quat = self.data.sensordata[quat_adr:quat_adr + 4].copy()  # w, x, y, z

        # Rotate world gravity [0, 0, -1] into body frame using quaternion
        # MuJoCo quaternion convention: (w, x, y, z)
        gravity_world = np.array([0.0, 0.0, -1.0])
        proj_grav = self._rotate_vector_by_quat_inv(gravity_world, quat)

        return ang_vel.astype(np.float32), proj_grav.astype(np.float32)

    def get_height_measurements(self) -> np.ndarray:
        """
        Return 132 processed heightmap observation values.

        This matches the training scandot formula:
            clip(base_z - 0.3 - measured_terrain_height, -1, 1)
        """
        base_pos, yaw = self._base_pos_yaw()
        c = np.cos(yaw)
        s = np.sin(yaw)
        local_xy = self.height_points[:, :2]
        world_xy = np.empty_like(local_xy)
        world_xy[:, 0] = c * local_xy[:, 0] - s * local_xy[:, 1] + base_pos[0]
        world_xy[:, 1] = s * local_xy[:, 0] + c * local_xy[:, 1] + base_pos[1]

        measured = np.zeros(self.height_points.shape[0], dtype=np.float32)
        ray_origin = np.zeros(3, dtype=np.float64)
        ray_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id = np.zeros(1, dtype=np.int32)
        ray_start_z = float(base_pos[2] + 5.0)

        old_groups = self.model.geom_group[self.robot_geom_ids].copy()
        self.model.geom_group[self.robot_geom_ids] = 5
        try:
            for i, xy in enumerate(world_xy):
                ray_origin[:] = [float(xy[0]), float(xy[1]), ray_start_z]
                dist = mujoco.mj_ray(
                    self.model,
                    self.data,
                    ray_origin,
                    ray_vec,
                    self.raycast_geomgroup,
                    1,
                    self.raycast_bodyexclude,
                    geom_id,
                )
                if dist >= 0.0:
                    measured[i] = np.float32(ray_start_z - dist)
        finally:
            self.model.geom_group[self.robot_geom_ids] = old_groups

        heights = np.clip(
            float(base_pos[2]) - self.height_bias - measured,
            self.height_clip_min,
            self.height_clip_max,
        ).astype(np.float32)
        self.last_heightmap_values = heights.copy()
        self.last_heightmap_world_points = np.column_stack([world_xy, measured]).astype(np.float32)
        return heights

    def _base_pos_yaw(self) -> tuple:
        base_pos = self.data.qpos[:3].copy().astype(np.float32)
        _, _, yaw = quat_wxyz_to_roll_pitch_yaw(self.data.qpos[3:7])
        return base_pos, float(yaw)

    def update_goal_yaw(self):
        if self.goals is None or len(self.goals) == 0:
            self.goal_yaw[:] = 0.0
            return

        base_pos, yaw = self._base_pos_yaw()
        threshold = float(self.cfg.get('next_goal_threshold', 0.2))
        reach_delay = float(self.cfg.get('reach_goal_delay', 0.1))
        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        cur_goal = self.goals[goal_idx]
        if np.linalg.norm(base_pos[:2] - cur_goal[:2]) < threshold:
            self.reach_goal_timer += self.control_dt
            if self.reach_goal_timer > reach_delay and self.cur_goal_idx < len(self.goals) - 1:
                self.cur_goal_idx += 1
                self.reach_goal_timer = 0.0
        else:
            self.reach_goal_timer = 0.0

        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        next_idx = min(goal_idx + 1, len(self.goals) - 1)
        cur_goal = self.goals[goal_idx]
        next_goal = self.goals[next_idx]
        cur_vec = cur_goal[:2] - base_pos[:2]
        next_vec = next_goal[:2] - base_pos[:2]
        cur_yaw = np.arctan2(cur_vec[1], cur_vec[0])
        next_yaw = np.arctan2(next_vec[1], next_vec[0])
        self.goal_yaw[:] = [0.0, wrap_to_pi(cur_yaw - yaw), wrap_to_pi(next_yaw - yaw)]

    @staticmethod
    def _rotate_vector_by_quat_inv(v, q):
        """Rotate vector v by inverse of quaternion q (w,x,y,z)."""
        w, x, y, z = q
        # Correct Rotation matrix from quaternion (w, x, y, z)
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        # Transpose to get inverse rotation (world -> body)
        return R.T @ v

    def step_sim(self):
        """Step MuJoCo simulation with PD control (per sub-step, matching IsaacGym)."""
        with self.cmd_lock:
            target = self.target_pos.copy()
            kp = self.target_kp.copy()
            kd = self.target_kd.copy()

        # Step simulation (DECIMATION sub-steps)
        # PD torque is recomputed at EVERY sub-step using latest dof_pos/dof_vel
        # This matches IsaacGym's _compute_torques() called inside the decimation loop
        for _ in range(self.decimation):
            cur_pos = self.get_joint_pos()
            cur_vel = self.get_joint_vel()
            tau = kp * (target - cur_pos) - kd * cur_vel
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            for i in range(NUM_JOINTS):
                self.data.ctrl[self.actuator_idx[i]] = tau[i]
            mujoco.mj_step(self.model, self.data)

    def publish_state(self):
        """Publish joint states and IMU data."""
        pos = self.get_joint_pos()
        vel = self.get_joint_vel()
        ang_vel, proj_grav = self.get_imu_data()
        heights = self.get_height_measurements()
        self.update_goal_yaw()

        # /mujoco/joint_state: 24 floats (12 pos + 12 vel)
        state_msg = Float32MultiArray()
        state_msg.data = pos.tolist() + vel.tolist()
        self.pub_joint_state.publish(state_msg)

        # /fast_livo2/state6_imu_prop : 6 floats (3 ang_vel + 3 proj_grav)
        imu_msg = Float32MultiArray()
        imu_msg.data = ang_vel.tolist() + proj_grav.tolist()
        self.pub_imu.publish(imu_msg)

        height_msg = Float32MultiArray()
        height_msg.layout.dim = [
            MultiArrayDimension(label='x', size=int(self.height_points_x.size),
                                stride=int(heights.size)),
            MultiArrayDimension(label='y', size=int(self.height_points_y.size),
                                stride=int(self.height_points_y.size)),
        ]
        height_msg.data = heights.tolist()
        self.pub_height.publish(height_msg)

        goal_msg = Float32MultiArray()
        goal_msg.data = self.goal_yaw.tolist()
        self.pub_goal_yaw.publish(goal_msg)

        # /joint_states: for RViz
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = self.joint_names
        js_msg.position = pos.astype(float).tolist()
        js_msg.velocity = vel.astype(float).tolist()
        js_msg.effort = [0.0] * NUM_JOINTS
        self.pub_rviz_joint.publish(js_msg)

    def update_viewer_markers(self, viewer):
        if not (self.visualize_heightmap or self.visualize_goals or self.visualize_goal_dirs):
            return
        scene = viewer.user_scn
        scene.ngeom = 0
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        if self.visualize_heightmap:
            self._add_heightmap_markers(scene, mat)
        if self.visualize_goals:
            self._add_goal_markers(scene, mat)
        if self.visualize_goal_dirs:
            self._add_goal_direction_lines(scene, mat)

    def _add_heightmap_markers(self, scene, mat):
        if self.last_heightmap_world_points is None:
            return
        size = np.array([self.heightmap_marker_size] * 3, dtype=np.float64)
        values = self.last_heightmap_values
        for idx, point in enumerate(self.last_heightmap_world_points):
            if scene.ngeom >= scene.maxgeom:
                break
            value = 0.0 if values is None else float(values[idx])
            normalized = (value - self.height_clip_min) / max(self.height_clip_max - self.height_clip_min, 1e-6)
            normalized = float(np.clip(normalized, 0.0, 1.0))
            rgba = np.array([normalized, 0.2, 1.0 - normalized, 1.0], dtype=np.float32)
            pos = np.array([point[0], point[1], point[2] + self.heightmap_marker_size], dtype=np.float64)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba,
            )
            scene.ngeom += 1

    def _add_goal_markers(self, scene, mat):
        if self.goals is None or len(self.goals) == 0:
            return
        size = np.array([self.goal_marker_size] * 3, dtype=np.float64)
        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        next_idx = min(goal_idx + 1, len(self.goals) - 1)
        for idx, goal in enumerate(self.goals):
            if scene.ngeom >= scene.maxgeom:
                break
            if idx < goal_idx:
                rgba = np.array([0.35, 0.35, 0.35, 0.35], dtype=np.float32)
            elif idx == goal_idx:
                rgba = np.array([0.1, 1.0, 0.2, 1.0], dtype=np.float32)
            elif idx == next_idx:
                rgba = np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32)
            else:
                rgba = np.array([0.1, 0.55, 1.0, 0.75], dtype=np.float32)
            pos = np.array([goal[0], goal[1], goal[2] + self.goal_marker_size], dtype=np.float64)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size * (1.35 if idx == goal_idx else 1.0),
                pos,
                mat,
                rgba,
            )
            scene.ngeom += 1

    def _add_goal_direction_lines(self, scene, mat):
        if self.goals is None or len(self.goals) == 0:
            return
        base_pos, _ = self._base_pos_yaw()
        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        next_idx = min(goal_idx + 1, len(self.goals) - 1)
        base = np.array([base_pos[0], base_pos[1], base_pos[2]], dtype=np.float64)
        current_goal = np.array(
            [self.goals[goal_idx, 0], self.goals[goal_idx, 1], self.goals[goal_idx, 2] + self.goal_marker_size],
            dtype=np.float64,
        )
        next_goal = np.array(
            [self.goals[next_idx, 0], self.goals[next_idx, 1], self.goals[next_idx, 2] + self.goal_marker_size],
            dtype=np.float64,
        )
        self._add_goal_line(scene, mat, base, current_goal, np.array([1.0, 0.35, 0.25, 1.0], dtype=np.float32))
        self._add_goal_line(scene, mat, base, next_goal, np.array([0.0, 1.0, 0.5, 1.0], dtype=np.float32))

    def _add_goal_line(self, scene, mat, start, end, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        if np.linalg.norm(end[:2] - start[:2]) < 1e-5:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            mat,
            rgba,
        )
        geom = scene.geoms[scene.ngeom]
        if hasattr(mujoco, 'mjv_connector'):
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                self.goal_dir_marker_radius,
                start,
                end,
            )
        else:
            mujoco.mjv_makeConnector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                self.goal_dir_marker_radius,
                float(start[0]),
                float(start[1]),
                float(start[2]),
                float(end[0]),
                float(end[1]),
                float(end[2]),
            )
        scene.ngeom += 1

    def run(self):
        """Main simulation loop with MuJoCo viewer."""
        self.get_logger().info('Starting MuJoCo simulation with viewer...')
        self.get_logger().info('Waiting for /mujoco/joint_cmd from deploy_node...')
        if self.pingpong_mode:
            self.get_logger().info(
                'Ping-pong mode enabled: publish state -> wait cmd -> step physics (no wall-clock rate control).')

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count = 0
            while viewer.is_running() and rclpy.ok():
                if self.pingpong_mode:
                    # Process callbacks and publish latest state snapshot.
                    rclpy.spin_once(self, timeout_sec=0)
                    self.publish_state()
                    self.update_viewer_markers(viewer)
                    viewer.sync()

                    # Wait until controller sends next command.
                    while viewer.is_running() and rclpy.ok():
                        if self.cmd_event.wait(timeout=0.001):
                            break
                        rclpy.spin_once(self, timeout_sec=0)

                    if not viewer.is_running() or not rclpy.ok():
                        break

                    self.cmd_event.clear()
                    self.step_sim()
                else:
                    loop_start = time.time()

                    # Process ROS2 callbacks
                    rclpy.spin_once(self, timeout_sec=0)

                    # Step simulation
                    self.step_sim()

                    # Publish sensor data
                    self.publish_state()

                    # Update viewer
                    self.update_viewer_markers(viewer)
                    viewer.sync()

                    # Rate control
                    elapsed = time.time() - loop_start
                    sleep_time = self.control_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # Print status periodically
                step_count += 1
                if step_count % 50 == 0:  # Every 50 simulation control steps
                    pos = self.get_joint_pos()
                    self.get_logger().info(
                        f'Step {step_count}: base_z={self.data.qpos[2]:.3f} '
                        f'q0={pos[0]:.3f} q1={pos[1]:.3f} q2={pos[2]:.3f}')

        self.get_logger().info('MuJoCo viewer closed. Shutting down.')


def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-config', default='',
                        help='Path to robot yaml config')
    args, _ = parser.parse_known_args()

    # Find package directory (source workspace preferred)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)  # deploy_cpp/

    cfg_path = args.robot_config
    if not cfg_path:
        cfg_path = os.path.join(pkg_dir, 'config', 'robots', 'mybot_v3_parkour.yaml')
    if not os.path.exists(cfg_path):
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('deploy_cpp')
            if not args.robot_config:
                cfg_path = os.path.join(pkg_share, 'config', 'robots', 'mybot_v3_parkour.yaml')
        except Exception:
            pass

    if not os.path.exists(cfg_path):
        print(f'ERROR: Cannot find robot yaml config at {cfg_path}')
        sys.exit(1)

    cfg = load_robot_config(cfg_path)
    xml_path = resolve_path(pkg_dir, cfg['mujoco_xml_relpath'])
    if not os.path.exists(xml_path):
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('deploy_cpp')
            xml_path = resolve_path(pkg_share, cfg['mujoco_xml_relpath'])
        except Exception:
            pass

    if not os.path.exists(xml_path):
        print(f"ERROR: Cannot find MuJoCo XML from config: {cfg['mujoco_xml_relpath']}")
        sys.exit(1)

    print(f"[mujoco_sim_node] robot={cfg.get('robot_name', 'unknown')} cfg={cfg_path}")
    print(f"[mujoco_sim_node] xml={xml_path}")

    node = MujocoSimNode(xml_path, cfg)

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
