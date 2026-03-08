import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .observations import orange_grasped, put_orange_to_plate


def ee_to_nearest_orange_reward(
    env: ManagerBasedRLEnv,
    orange_cfgs: list[SceneEntityCfg],
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Shaped reward: exp(-5*dist) to nearest orange, to guide EE toward oranges."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 1, :]  # jaw position (num_envs, 3)
    dists = []
    for orange_cfg in orange_cfgs:
        orange: RigidObject = env.scene[orange_cfg.name]
        dist = torch.linalg.vector_norm(orange.data.root_pos_w - ee_pos, dim=1)
        dists.append(dist)
    min_dist = torch.stack(dists, dim=1).min(dim=1).values
    return scale * torch.exp(-5.0 * min_dist)


def orange_grasped_reward(
    env: ManagerBasedRLEnv,
    orange_cfg: SceneEntityCfg,
    scale: float = 1.0,
) -> torch.Tensor:
    """Binary reward for each step an orange is grasped."""
    grasped = orange_grasped(env, object_cfg=orange_cfg)
    return scale * grasped.float()


def orange_placed_reward(
    env: ManagerBasedRLEnv,
    orange_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
    scale: float = 2.0,
) -> torch.Tensor:
    """Binary reward for each step an orange is placed on the plate."""
    placed = put_orange_to_plate(env, object_cfg=orange_cfg, plate_cfg=plate_cfg)
    return scale * placed.float()


def oranges_on_plate_bonus(
    env: ManagerBasedRLEnv,
    orange_cfgs: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
    scale: float = 10.0,
) -> torch.Tensor:
    """One-time bonus when all oranges are on the plate (no rest-pose requirement)."""
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_height = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for orange_cfg in orange_cfgs:
        orange: RigidObject = env.scene[orange_cfg.name]
        orange_x = orange.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        orange_y = orange.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        orange_height = orange.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        done = torch.logical_and(done, orange_x < plate_x + x_range[1])
        done = torch.logical_and(done, orange_x > plate_x + x_range[0])
        done = torch.logical_and(done, orange_y < plate_y + y_range[1])
        done = torch.logical_and(done, orange_y > plate_y + y_range[0])
        done = torch.logical_and(done, orange_height < plate_height + height_range[1])
        done = torch.logical_and(done, orange_height > plate_height + height_range[0])

    return scale * done.float()
