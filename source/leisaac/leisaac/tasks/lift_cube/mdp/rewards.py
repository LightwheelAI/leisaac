import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

# Fingertip offsets in each body's local frame, calibrated from USD collision mesh geometry
_JAW_TIP_OFFSET = torch.tensor([0.0, -0.05, 0.02])
_GRIPPER_TIP_OFFSET = torch.tensor([-0.012, 0.0, -0.08])


def _tcp_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """TCP position: midpoint between jaw tip and gripper tip in world frame. (num_envs, 3)"""
    robot: Articulation = env.scene[robot_cfg.name]
    jaw_idx = robot.data.body_names.index("jaw")
    gripper_idx = robot.data.body_names.index("gripper")

    jaw_pos = robot.data.body_pos_w[:, jaw_idx, :]
    jaw_quat = robot.data.body_quat_w[:, jaw_idx, :]
    gripper_pos = robot.data.body_pos_w[:, gripper_idx, :]
    gripper_quat = robot.data.body_quat_w[:, gripper_idx, :]

    device = jaw_pos.device
    N = jaw_pos.shape[0]
    jaw_offset = _JAW_TIP_OFFSET.to(device).unsqueeze(0).expand(N, -1)
    gripper_offset = _GRIPPER_TIP_OFFSET.to(device).unsqueeze(0).expand(N, -1)

    jaw_tip = jaw_pos + quat_apply(jaw_quat, jaw_offset)
    gripper_tip = gripper_pos + quat_apply(gripper_quat, gripper_offset)
    return (jaw_tip + gripper_tip) / 2.0


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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    k: float = 5.0,
) -> torch.Tensor:
    """Reaching reward: 1 - tanh(k * dist) from TCP (grasp midpoint) to cube center. Range [0, 1]."""
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _tcp_pos(env, robot_cfg), dim=1)
    return 1.0 - torch.tanh(k * dist)


def cube_height_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.046,
    k: float = 5.0,
) -> torch.Tensor:
    """Height reward: tanh(k * max(h - min_height, 0)). Zero below min_height, monotonically increasing above. Range [0, 1]."""
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    h = torch.clamp(height_above_base - min_height, min=0.0)
    return torch.tanh(k * h)


def cube_success_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """One-time large bonus when cube reaches the success height threshold."""
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    return (height_above_base >= height_threshold).float()
