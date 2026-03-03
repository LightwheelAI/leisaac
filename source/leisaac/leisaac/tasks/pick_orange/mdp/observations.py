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
