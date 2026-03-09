import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer


def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 1, :]  # jaw position (num_envs, 3)


def _cube_pos(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w  # (num_envs, 3)


def _robot_base_height(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, base_name: str = "base") -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(base_name)
    return robot.data.body_pos_w[:, base_index, 2]  # (num_envs,)


def ee_to_cube_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    k: float = 5.0,
) -> torch.Tensor:
    """Tanh reaching reward: 1 - tanh(k * dist), range [0, 1]."""
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    return 1.0 - torch.tanh(k * dist)


def _is_grasped(env: ManagerBasedRLEnv, contact_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Binary grasp detection: jaw in contact with cube AND gripper closed. Shape: (num_envs,)."""
    contact_sensor: ContactSensor = env.scene[contact_cfg.name]
    # net_forces_w: (num_envs, num_bodies, 3) — here num_bodies=1 (jaw)
    forces = contact_sensor.data.net_forces_w[:, 0, :]
    in_contact = (torch.linalg.vector_norm(forces, dim=1) > 0.5).float()

    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    gripper_closed = torch.sigmoid(20.0 * (0.7 - gripper_pos))

    return in_contact * gripper_closed


def cube_grasped_reward(
    env: ManagerBasedRLEnv,
    contact_cfg: SceneEntityCfg = SceneEntityCfg("jaw_contact"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Grasp reward: jaw physically contacts cube AND gripper is closed. Range [0, 1]."""
    return _is_grasped(env, contact_cfg, robot_cfg)


def cube_height_if_grasped(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("jaw_contact"),
    target_height: float = 0.20,
) -> torch.Tensor:
    """Exponential height reward, only when jaw is in contact with cube. Range [0, 1].

    exp(-2 * |h - target|) peaks at target_height.
    """
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    height_rew = torch.exp(-2.0 * torch.abs(height_above_base - target_height))
    return height_rew * _is_grasped(env, contact_cfg, robot_cfg)


def cube_success_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """One-time large bonus when cube reaches the success height threshold."""
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    return (height_above_base >= height_threshold).float()
