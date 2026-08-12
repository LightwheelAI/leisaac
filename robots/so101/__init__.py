"""SO-101 digital twin boundary.

This module is the single entry point for SO-101 robot-specific knowledge:
joint names/order, joint limits, actuator configuration, gripper, default pose,
and robot-side observation extraction.

It MUST NOT contain any Pi0.5 / GR00T / LeRobot-training specific code. Those
layers depend on this one, never the reverse.

The authoritative joint configuration lives in the upstream LeIsaac module
``leisaac.assets.robots.lerobot`` (ArticulationCfg SO101_FOLLOWER_CFG) and the
joint-limit tables in ``leisaac.assets.robots.lerobot``. This module re-exports
them so the rest of the repository has one stable import surface.

Runtime status:
  SO-101 asset / joint config / actuators / gripper  -> VERIFIED in source
  Spawn / actuation in live sim                      -> BLOCKED by Isaac Sim
                                                          omni.isaac.ml_archive
                                                          extension defect
                                                          (see docs/upstream.md / README)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Canonical SO-101 joint order (MUST match leisaac.assets.robots.lerobot).
# 5 arm joints + 1 gripper = 6-DoF action/state vector.
# ---------------------------------------------------------------------------
SO101_JOINT_NAMES: Final[tuple[str, ...]] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Degrees (written in the SO-101 follower USD).
SO101_USD_JOINT_LIMITS_DEG: Final[dict[str, tuple[float, float]]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# Motor limits in the normalized LeRobot space (what Pi0.5 is trained on).
SO101_MOTOR_LIMITS: Final[dict[str, tuple[float, float]]] = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

SO101_ACTION_DIM: Final[int] = 6
SO101_STATE_DIM: Final[int] = 6

# Default rest pose in motor-normalized degrees (from
# SO101_FOLLOWER_REST_POSE_RANGE midpoint in leisaac source).
SO101_REST_POSE_DEG: Final[dict[str, float]] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": -10.0,
}

# Default camera keys used by the SO-101 follower rig.
# (These match the LiftCube / template camera prims in LeIsaac.)
SO101_CAMERA_KEYS: Final[tuple[str, ...]] = ("front", "wrist")


@dataclass(frozen=True)
class SO101RobotSpec:
    """Immutable description of the SO-101 follower digital twin."""

    joint_names: tuple[str, ...] = SO101_JOINT_NAMES
    action_dim: int = SO101_ACTION_DIM
    state_dim: int = SO101_STATE_DIM
    usd_joint_limits_deg: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(SO101_USD_JOINT_LIMITS_DEG)
    )
    motor_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(SO101_MOTOR_LIMITS)
    )
    rest_pose_deg: dict[str, float] = field(
        default_factory=lambda: dict(SO101_REST_POSE_DEG)
    )
    camera_keys: tuple[str, ...] = SO101_CAMERA_KEYS

    @property
    def joint_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.joint_names)}


# A ready-to-use singleton spec.
SO101_SPEC = SO101RobotSpec()


def load_isaac_articulation_cfg():
    """Return the authoritative IsaacLab ArticulationCfg for SO-101.

    Delegates to upstream LeIsaac so there is one source of truth. Requires the
    Isaac Sim runtime (``omni``) to import; call only inside a launched app.
    """
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG

    return SO101_FOLLOWER_CFG
