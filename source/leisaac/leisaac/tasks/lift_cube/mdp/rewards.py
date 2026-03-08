import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _gripper_root_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, body_name: str = "gripper") -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    body_index = robot.data.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_index, :]  # (num_envs, 3)


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
    """Reaching reward: closer gripper body is to 10cm above cube center, the higher. Range [0, 1]."""
    target_pos = _cube_pos(env, cube_cfg).clone()
    target_pos[:, 2] += 0.10
    dist = torch.linalg.vector_norm(target_pos - _gripper_root_pos(env, robot_cfg), dim=1)
    return 1.0 - torch.tanh(k * dist)


def _is_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Grasp detection: contact on both sides + gripper closed + upward support force.

    Four conditions (all soft via sigmoid):
      1. jaw_contact force > 0.5 N  (sensor filtered to cube-only)
      2. gripper_contact force > 0.5 N  (sensor filtered to cube-only)
      3. gripper joint < 0.5 rad (gripper actually closed)
      4. contact force Z < 0 on both sensors: cube pulls gripper down via friction,
         consistent with supporting cube weight. Rejects top-smash (Z > 0).
    Shape: (num_envs,), range [0, 1].
    """
    jaw_sensor: ContactSensor = env.scene["jaw_contact"]
    gripper_sensor: ContactSensor = env.scene["gripper_contact"]

    jaw_force = torch.linalg.vector_norm(jaw_sensor.data.net_forces_w[:, 0, :], dim=1)
    gripper_force = torch.linalg.vector_norm(gripper_sensor.data.net_forces_w[:, 0, :], dim=1)

    jaw_contact = torch.sigmoid(5.0 * (jaw_force - 0.5))
    gripper_contact = torch.sigmoid(5.0 * (gripper_force - 0.5))

    # Gripper closed constraint: joint≈0.2 (closed) → 1, joint≈1.0 (open) → 0
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    gripper_closed = torch.sigmoid(10.0 * (0.5 - gripper_pos))

    # Upward support: sensor Z force < 0 means cube is pulling gripper down (friction grip).
    # Top-smash produces Z > 0 (cube pushes gripper up). Scale 2.0 gives soft threshold near 0N.
    jaw_z = jaw_sensor.data.net_forces_w[:, 0, 2]
    grip_z = gripper_sensor.data.net_forces_w[:, 0, 2]
    upward_support = torch.sigmoid(-2.0 * jaw_z) * torch.sigmoid(-2.0 * grip_z)

    return jaw_contact * gripper_contact * gripper_closed * upward_support


def cube_grasped_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Grasp reward: contact + closed + upward friction support. Range [0, 1]."""
    return _is_grasped(env, robot_cfg=robot_cfg)


def cube_height_if_grasped(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.20,
    sharpness: float = 2.0,
    lift_threshold: float = 0.08,
) -> torch.Tensor:
    """Exponential height reward, only when grasped AND cube lifted above lift_threshold.

    lift_threshold filters out the resting-on-table baseline (~0.046m). Range [0, 1].
    """
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    height_rew = torch.exp(-sharpness * torch.abs(height_above_base - target_height))
    lifted = (height_above_base > lift_threshold).float()
    return height_rew * _is_grasped(env, robot_cfg=robot_cfg) * lifted


def gripper_close_when_near(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    near_dist: float = 0.10,
    k: float = 50.0,
) -> torch.Tensor:
    """Reward gripper closure when gripper body is within near_dist of cube. Range [0, 1]."""
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _gripper_root_pos(env, robot_cfg), dim=1)
    near = torch.sigmoid(k * (near_dist - dist))

    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    closed_signal = torch.sigmoid(10.0 * (0.5 - gripper_pos))  # 1 when closed (joint≈0.2)

    return near * closed_signal


def cube_success_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """One-time large bonus when cube reaches the success height threshold."""
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    return (height_above_base >= height_threshold).float()
