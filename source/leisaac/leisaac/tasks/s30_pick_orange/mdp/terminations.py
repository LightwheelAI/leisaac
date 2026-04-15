from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from leisaac.utils.robot_utils import is_s30_at_rest_pose


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Task-done check for the S30 + Robotiq pick-orange task.

    Identical to the SO101 version except the rest-pose check uses S30 joint names.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_height = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    for orange_cfg in oranges_cfg:
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

    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    done = torch.logical_and(done, is_s30_at_rest_pose(joint_pos, joint_names))

    return done
