from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedEnv
from isaaclab.sensors import Camera
from leisaac.assets.robots.lerobot import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_REST_POSE_RANGE,
    SO101_FOLLOWER_USD_JOINT_LIMLITS,
)
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg


@dataclass
class StateFeatureItem:
    dtype: str = "float32"
    shape: tuple = (6,)
    names: list[str] = field(
        default_factory=lambda: ["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos"]
    )


@dataclass
class VideoFeatureItem:
    dtype: str = "video"
    shape: list = field(default_factory=lambda: [480, 640, 3])  # [h, w, c]
    names: list[str] = field(default_factory=lambda: ["height", "width", "channels"])
    video_info: dict = field(
        default_factory=lambda: {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        }
    )


def build_feature_from_env(env: ManagerBasedEnv | DirectRLEnv, dataset_cfg: LeRobotDatasetCfg) -> dict:
    """
    Build the feature from the environment.
    """
    features = {}

    default_feature_joint_names = env.cfg.default_feature_joint_names
    if isinstance(env, ManagerBasedEnv):
        action_dim = env.action_manager.total_action_dim
    else:
        action_dim = env.actions.shape[-1]

    if action_dim != len(default_feature_joint_names):
        # [A bit tricky, currently works because the action dimension matches the joints only when we use leader control]
        action_joint_names = [f"dim_{index}" for index in range(action_dim)]
    else:
        action_joint_names = default_feature_joint_names
    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(action_dim,), names=action_joint_names))
    features["observation.state"] = asdict(
        StateFeatureItem(dtype="float32", shape=(len(default_feature_joint_names),), names=default_feature_joint_names)
    )

    for camera_key, camera_sensor in env.scene.sensors.items():
        if isinstance(camera_sensor, Camera):
            height, width = camera_sensor.image_shape
            video_feature_item = VideoFeatureItem(
                dtype="video", shape=[height, width, 3], names=["height", "width", "channels"]
            )
            video_feature_item.video_info["video.height"] = height
            video_feature_item.video_info["video.width"] = width
            video_feature_item.video_info["video.fps"] = dataset_cfg.fps
            features[f"observation.images.{camera_key}"] = asdict(video_feature_item)

    return features


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


def convert_lekiwi_wheel_action_robot2env(action: torch.Tensor, base_theta: torch.Tensor) -> torch.Tensor:
    """
    Convert the wheel action from robot to environment.

    Args:
        action: (N, 3) tensor in user command frame, [forward, left, rotate]. (m/s, m/s, rad/s)
        base_theta: (N,) tensor for robot base yaw (around its own z-axis) in world frame.

    Returns:
        (N, 3) tensor in world frame, [dx_world, dy_world, dtheta_body](m/s, m/s, rad/s), where translation
        is expressed in world XY axes and rotation remains in the robot/body frame (about its local z-axis).
    """

    cos_yaw = torch.cos(base_theta)
    sin_yaw = torch.sin(base_theta)

    # Convert user command frame to body frame
    dx_body = action[:, 1] * -1.0  # left(negative)
    dy_body = action[:, 0]  # forward(positive)
    dtheta_body = action[:, 2]  # rotate(positive)

    # Rotate body-frame translation into world frame
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body

    return torch.stack((dx_world, dy_world, dtheta_body), dim=-1)


def convert_lekiwi_wheel_action_env2robot(action: torch.Tensor | np.ndarray, base_theta: torch.Tensor) -> torch.Tensor:
    """
    Convert the wheel action from environment(world frame) back to user command frame.

    Args:
        action: (N, 3) tensor in world frame, [dx_world, dy_world, dtheta_body]. (m/s, m/s, rad/s)
        base_theta: (N,) tensor for robot base yaw (around its own z-axis) in world frame.

    Returns:
        (N, 3) tensor in user command frame, [forward, left, rotate]. (m/s, m/s, rad/s)
    """

    cos_yaw = torch.cos(base_theta)
    sin_yaw = torch.sin(base_theta)

    dx_world = action[:, 0]
    dy_world = action[:, 1]
    dtheta_body = action[:, 2]

    # Rotate world-frame translation back to body frame
    dx_body = cos_yaw * dx_world + sin_yaw * dy_world
    dy_body = -sin_yaw * dx_world + cos_yaw * dy_world

    # Convert body-frame to user command frame
    forward = dy_body  # forward(positive)
    left = -dx_body  # left(negative)
    rotate = dtheta_body

    # Apply small-value thresholding to avoid numerical noise
    eps = 1e-4
    forward = torch.where(torch.abs(forward) < eps, torch.zeros_like(forward), forward)
    left = torch.where(torch.abs(left) < eps, torch.zeros_like(left), left)
    rotate = torch.where(torch.abs(rotate) < eps, torch.zeros_like(rotate), rotate)

    return torch.stack((forward, left, rotate), dim=-1)
