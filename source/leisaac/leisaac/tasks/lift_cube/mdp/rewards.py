import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .observations import object_grasped


def ee_to_cube_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Shaped reward: exp(-5*dist) from EE jaw to cube center."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 1, :]
    cube: RigidObject = env.scene[cube_cfg.name]
    dist = torch.linalg.vector_norm(cube.data.root_pos_w - ee_pos, dim=1)
    return scale * torch.exp(-5.0 * dist)


def cube_grasped_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Binary reward for each step the cube is grasped."""
    grasped = object_grasped(env, object_cfg=cube_cfg)
    return scale * grasped.float()


def cube_height_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
    scale: float = 1.0,
) -> torch.Tensor:
    """Shaped reward proportional to how high the cube is above the robot base."""
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    robot_base_height = robot.data.body_pos_w[:, base_index, 2]
    height_above_base = (cube.data.root_pos_w[:, 2] - robot_base_height).clamp(min=0.0)
    return scale * height_above_base


def cube_lift_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
    height_threshold: float = 0.20,
    scale: float = 10.0,
) -> torch.Tensor:
    """Sparse bonus when cube is lifted above height_threshold."""
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    robot_base_height = robot.data.body_pos_w[:, base_index, 2]
    lifted = (cube.data.root_pos_w[:, 2] - robot_base_height) > height_threshold
    return scale * lifted.float()
