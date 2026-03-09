import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def orange_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check if an object(orange) is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def put_orange_to_plate(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check if an object(orange) is placed on the specified plate."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    orange: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    plate_x, plate_y = plate.data.root_pos_w[:, 0], plate.data.root_pos_w[:, 1]
    orange_x, orange_y = orange.data.root_pos_w[:, 0], orange.data.root_pos_w[:, 1]
    orange_in_plate_x = torch.logical_and(orange_x < plate_x + x_range[1], orange_x > plate_x + x_range[0])
    orange_in_plate_y = torch.logical_and(orange_y < plate_y + y_range[1], orange_y > plate_y + y_range[0])
    orange_in_plate = torch.logical_and(orange_in_plate_x, orange_in_plate_y)

    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    orange_pos = orange.data.root_pos_w
    pos_diff = torch.linalg.vector_norm(orange_pos - end_effector_pos, dim=1)
    ee_near_to_orange = pos_diff < diff_threshold

    gripper_open = robot.data.joint_pos[:, -1] > grasp_threshold

    placed = torch.logical_and(orange_in_plate, ee_near_to_orange)
    placed = torch.logical_and(placed, gripper_open)

    return placed


def oranges_pos_relative_to_ee(
    env: ManagerBasedRLEnv,
    orange_cfgs: list[SceneEntityCfg],
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Returns positions of all oranges relative to the EE jaw, flattened to (num_envs, 3*n_oranges)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 1, :]  # jaw position (num_envs, 3)
    relative_positions = []
    for orange_cfg in orange_cfgs:
        orange: RigidObject = env.scene[orange_cfg.name]
        relative_pos = orange.data.root_pos_w - ee_pos  # (num_envs, 3)
        relative_positions.append(relative_pos)
    return torch.cat(relative_positions, dim=1)  # (num_envs, 3*n_oranges)


def plate_pos_relative_to_ee(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Returns plate position relative to the EE jaw. (num_envs, 3)"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 1, :]  # jaw position
    plate: RigidObject = env.scene[plate_cfg.name]
    return plate.data.root_pos_w - ee_pos


def oranges_task_status(
    env: ManagerBasedRLEnv,
    orange_cfgs: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Returns placement status (0/1) for each orange. (num_envs, n_oranges)

    1 = orange is on the plate, 0 = not yet placed.
    Gives the policy task-progress information so it knows which oranges remain.
    """
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_z = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    statuses = []
    for orange_cfg in orange_cfgs:
        orange: RigidObject = env.scene[orange_cfg.name]
        ox = orange.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        oy = orange.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        oz = orange.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        on_plate = (
            (ox > plate_x + x_range[0])
            & (ox < plate_x + x_range[1])
            & (oy > plate_y + y_range[0])
            & (oy < plate_y + y_range[1])
            & (oz > plate_z + height_range[0])
            & (oz < plate_z + height_range[1])
        )
        statuses.append(on_plate.float().unsqueeze(1))
    return torch.cat(statuses, dim=1)  # (num_envs, n_oranges)
