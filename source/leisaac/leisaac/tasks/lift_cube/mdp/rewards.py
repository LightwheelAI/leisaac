import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 1, :]  # jaw position (num_envs, 3)


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
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Gaussian reward: peaks at 1.0 when EE is at cube, falls to ~0.37 at sigma distance.

    Using Gaussian kernel exp(-dist^2 / (2*sigma^2)) gives a true bell-curve gradient
    that provides strong signal when close and weak signal when far — better than exp(-k*dist).
    sigma=0.1 means half-reward at ~12cm.
    """
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    return torch.exp(-(dist**2) / (2 * sigma**2))


def ee_near_cube_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    sigma: float = 0.04,
) -> torch.Tensor:
    """Fine-grained Gaussian reward for EE being very close to cube (σ=4cm).

    Separate from ee_to_cube (σ=10cm) to provide a sharper gradient in the final approach.
    This reward is gripper-agnostic: the policy should approach with gripper open.
    """
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    return torch.exp(-(dist**2) / (2 * sigma**2))


def gripper_close_near_cube_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma_pos: float = 0.04,
) -> torch.Tensor:
    """Reward closing the gripper ONLY when EE is already near the cube.

    pos_gate × gripper_closed: the position gate (σ=4cm) ensures the closure reward
    only activates once the EE has arrived. This separates the approach phase (handled
    by ee_to_cube / ee_near_cube) from the grasp phase, respecting the temporal ordering:
    approach with open gripper → arrive → close gripper.

    With binary gripper: gripper_pos ∈ {1.0 (open), 0.4 (closed)}.
    gripper_closed = sigmoid(k*(0.7 - gripper_pos)) → 1.0 when closed, ~0 when open.
    """
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    pos_gate = torch.exp(-(dist**2) / (2 * sigma_pos**2))

    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    # sigmoid: gripper_pos=0.4(closed)→~1.0, gripper_pos=1.0(open)→~0.0
    gripper_closed = torch.sigmoid(20.0 * (0.7 - gripper_pos))

    return pos_gate * gripper_closed


def _update_hold_counter(
    env: ManagerBasedRLEnv,
    near: torch.Tensor,
    counter_name: str,
) -> torch.Tensor:
    """Increment per-env counter when near=True, reset to 0 otherwise.

    Also resets on episode boundaries (episode_length_buf <= 1).
    Returns the updated counter (num_envs,).
    """
    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    counter: torch.Tensor = getattr(env, counter_name)

    # Reset on episode boundary
    episode_reset = env.episode_length_buf <= 1
    counter = torch.where(episode_reset | ~near, torch.zeros_like(counter), counter + 1)
    setattr(env, counter_name, counter)
    return counter


def cube_success_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """One-time large bonus when cube reaches the success height threshold.

    Since the episode terminates immediately on success, this is effectively a terminal reward.
    """
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    return (height_above_base >= height_threshold).float()


def cube_stable_hold_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_dist_threshold: float = 0.08,
    min_hold_steps: int = 10,
    ground_clearance: float = 0.02,
) -> torch.Tensor:
    """Flat reward (1.0) whenever cube is stably held off the ground.

    Conditions: EE near cube for min_hold_steps consecutive steps AND cube height > ground_clearance.
    Gives a large constant reward as soon as stable hold is detected, regardless of exact height.
    """
    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    near = dist <= ee_dist_threshold
    hold_steps = _update_hold_counter(env, near, "_cube_stable_hold_steps")
    hold_gate = (hold_steps >= min_hold_steps).float()

    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    off_ground = (height_above_base > ground_clearance).float()

    return hold_gate * off_ground


def cube_height_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_height: float = 0.20,
    sigma: float = 0.08,
    ee_dist_threshold: float = 0.08,
    min_hold_steps: int = 10,
) -> torch.Tensor:
    """Gaussian height reward, only given after EE has been near cube for min_hold_steps consecutive steps.

    Prevents rewarding knocked/bounced cube: a brief contact won't trigger the reward.
    The hold counter resets whenever dist > ee_dist_threshold or the episode resets.
    """
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    height_rew = torch.exp(-((height_above_base - target_height) ** 2) / (2 * sigma**2))

    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    near = dist <= ee_dist_threshold
    hold_steps = _update_hold_counter(env, near, "_cube_height_hold_steps")
    hold_gate = (hold_steps >= min_hold_steps).float()

    return height_rew * hold_gate


def cube_lift_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    height_threshold: float = 0.20,
    sharpness: float = 20.0,
    ee_dist_threshold: float = 0.08,
    min_hold_steps: int = 10,
) -> torch.Tensor:
    """Sigmoid step bonus, only given after EE has been near cube for min_hold_steps consecutive steps.

    Same hold condition as cube_height_reward but uses a separate counter.
    """
    height_above_base = _cube_pos(env, cube_cfg)[:, 2] - _robot_base_height(env, robot_cfg)
    bonus = torch.sigmoid(sharpness * (height_above_base - height_threshold))

    dist = torch.linalg.vector_norm(_cube_pos(env, cube_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    near = dist <= ee_dist_threshold
    hold_steps = _update_hold_counter(env, near, "_cube_lift_hold_steps")
    hold_gate = (hold_steps >= min_hold_steps).float()

    return bonus * hold_gate
