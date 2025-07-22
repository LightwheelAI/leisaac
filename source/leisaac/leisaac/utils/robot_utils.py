
import torch

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_REST_POSE_RANGE


def is_so101_at_rest_pose(joint_pos: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """
    Check if the robot is in the rest pose.
    """
    is_reset = torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
    reset_pose_range = SO101_FOLLOWER_REST_POSE_RANGE
    joint_pos = joint_pos / torch.pi * 180.0  # change to degree
    for joint_name, (min_pos, max_pos) in reset_pose_range.items():
        joint_idx = joint_names.index(joint_name)
        is_reset = torch.logical_and(is_reset, torch.logical_and(joint_pos[:, joint_idx] > min_pos, joint_pos[:, joint_idx] < max_pos))
    return is_reset