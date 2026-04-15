"""S30 + Robotiq 2F-140 specific event functions.

Replaces the ActionGraph (removed to avoid eENABLE_DIRECT_GPU_API conflicts).

Robotiq 2F-140 mechanism (USD joint names are authoritative, image of stage tree confirmed):

  Joint structure (10 PhysicsRevolute joints):
    finger_joint               : 0-45°, main driver (controlled by action manager)
    right_outer_knuckle_joint  : 0-45°, mirrored by this event (ratio +1, same direction)
    left_inner_knuckle_joint   : no limit, PhysX gear-constrained
    right_inner_knuckle_joint  : no limit, PhysX gear-constrained
    left_outer_finger_joint    : 0-180°, PhysX gear-constrained
    right_outer_finger_joint   : 0-180°, PhysX gear-constrained
    left_inner_finger_joint    : no limit, PhysX gear-constrained
    right_inner_finger_joint   : no limit, PhysX gear-constrained
    left_inner_finger_pad_joint: -45~+45°, PhysX gear-constrained (no explicit actuator)
    right_inner_finger_pad_joint:-45~+45°, PhysX gear-constrained (no explicit actuator)

  NOTE: left_inner_finger_pad_joint ≠ left_inner_knuckle_joint — they are distinct
  joints in the stage.  The ActionGraph drove ONLY finger_joint + right_outer_knuckle_joint
  (both at ratio +1, 0-45°).  All 4-bar joints are gear-constrained in physics_edit.usd.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

_MIMIC_JOINTS: list[tuple[str, float]] = [
    # ActionGraph drove both finger_joint and right_outer_knuckle_joint to the SAME
    # 0-45° value (ratio +1).  The right side joint axis is already defined to close
    # in the positive direction, matching finger_joint convention.
    ("right_outer_knuckle_joint",    +1.0),
    # inner_finger_pad_joint is NVIDIA-specific (not in standard URDF mimic list).
    # No PhysX gear constraint → must be tracked explicitly.
    # When finger closes by θ, pad must rotate -θ to stay parallel (range -45~+45°).
    ("left_inner_finger_pad_joint",  1.0),
    ("right_inner_finger_pad_joint", 1.0),
]

_joint_cache: dict[str, tuple[int, list[tuple[int, float]]]] = {}


def track_robotiq_mimic_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Mirror right_outer_knuckle_joint to finger_joint every physics step (ratio +1)."""
    robot: Articulation = env.scene[asset_cfg.name]

    cache_key = asset_cfg.name
    if cache_key not in _joint_cache:
        finger_ids, _ = robot.find_joints("finger_joint")
        if not finger_ids:
            return
        mimic_list: list[tuple[int, float]] = []
        for joint_name, ratio in _MIMIC_JOINTS:
            try:
                ids, _ = robot.find_joints(joint_name)
                if ids:
                    mimic_list.append((ids[0], ratio))
            except ValueError:
                pass
        _joint_cache[cache_key] = (finger_ids[0], mimic_list)

    finger_idx, mimic_list = _joint_cache[cache_key]
    if not mimic_list:
        return

    finger_pos = robot.data.joint_pos[:, finger_idx]
    for joint_idx, ratio in mimic_list:
        robot.set_joint_position_target(
            (finger_pos * ratio).unsqueeze(-1), joint_ids=[joint_idx]
        )
