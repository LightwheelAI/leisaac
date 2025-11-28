import numpy as np
import torch
from leisaac.assets.robots.lerobot import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_REST_POSE_RANGE,
    SO101_FOLLOWER_USD_JOINT_LIMLITS,
)


def is_so101_at_rest_pose(joint_pos: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """
    Check if the robot is in the rest pose.
    """
    is_reset = torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
    reset_pose_range = SO101_FOLLOWER_REST_POSE_RANGE
    joint_pos = joint_pos / torch.pi * 180.0  # change to degree
    for joint_name, (min_pos, max_pos) in reset_pose_range.items():
        joint_idx = joint_names.index(joint_name)
        is_reset = torch.logical_and(
            is_reset, torch.logical_and(joint_pos[:, joint_idx] > min_pos, joint_pos[:, joint_idx] < max_pos)
        )
    return is_reset


def convert_leisaac_action_to_lerobot(action: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert the action from LeIsaac to Lerobot. Just convert value, not include the format.
    """
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed_action = np.zeros_like(action)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    motor_limits = SO101_FOLLOWER_MOTOR_LIMITS
    action = action / torch.pi * 180.0  # convert to degree

    for idx, joint_name in enumerate(joint_limits):
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_degree = action[:, idx] - joint_limit_range[0]
        processed_action[:, idx] = joint_degree / joint_range * motor_range + motor_limit_range[0]

    return processed_action


def convert_lerobot_action_to_leisaac(action: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert the action from Lerobot to LeIsaac. Just convert value, not include the format.
    """
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed_action = np.zeros_like(action)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

    for idx, joint_name in enumerate(joint_limits):
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_degree = action[:, idx] - motor_limit_range[0]
        processed_degree = motor_degree / motor_range * joint_range + joint_limit_range[0]
        processed_radius = processed_degree / 180.0 * torch.pi  # convert to radian
        processed_action[:, idx] = processed_radius

    return processed_action
