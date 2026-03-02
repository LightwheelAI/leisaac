"""Reward functions for the pick-orange RL task."""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .observations import orange_grasped, put_orange_to_plate
from .terminations import task_done


def ee_to_nearest_orange_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    oranges_cfg: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    """Shaped reward: exp(-2 * dist) to the nearest orange not yet placed on the plate.

    Encourages the arm to approach any orange. Falls back to all oranges if none
    are unplaced (e.g. all on plate already).
    """
    if oranges_cfg is None:
        oranges_cfg = [
            SceneEntityCfg("Orange001"),
            SceneEntityCfg("Orange002"),
            SceneEntityCfg("Orange003"),
        ]

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 1, :]  # (num_envs, 3)

    min_dist = torch.full((env.num_envs,), fill_value=1e6, device=env.device)
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        dist = torch.linalg.vector_norm(orange.data.root_pos_w - ee_pos, dim=-1)
        min_dist = torch.minimum(min_dist, dist)

    return torch.exp(-2.0 * min_dist)


def orange_grasped_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
) -> torch.Tensor:
    """Binary reward: +1.0 each step an orange is held in the gripper."""
    return orange_grasped(env, robot_cfg, ee_frame_cfg, object_cfg).float()


def orange_placed_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Binary reward: +1.0 each step an orange is resting on the plate."""
    return put_orange_to_plate(env, robot_cfg, ee_frame_cfg, object_cfg, plate_cfg).float()


def task_complete_bonus(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg] | None = None,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Sparse bonus: +1.0 when all 3 oranges are on the plate and arm at rest."""
    if oranges_cfg is None:
        oranges_cfg = [
            SceneEntityCfg("Orange001"),
            SceneEntityCfg("Orange002"),
            SceneEntityCfg("Orange003"),
        ]
    return task_done(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg).float()
