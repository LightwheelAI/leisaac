"""RL environment configuration for the pick-orange task.

Uses joint-position control (no IK) with proprioceptive-only observations
so the policy can be trained with rsl_rl PPO.

Action space (6D):  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
Observation space (28D):
    joint_pos          (6)
    joint_vel          (6)
    ee_frame_state     (7)  — EE position (3) + quaternion (4) in robot base frame
    oranges_pos_rel    (9)  — 3 oranges × 3D position relative to EE
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .pick_orange_env_cfg import PickOrangeEnvCfg
from . import mdp


@configclass
class PickOrangeRLObsCfg:
    """Proprioceptive-only observation config for RL (28D, concatenated)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        oranges_pos_relative = ObsTerm(
            func=mdp.oranges_pos_relative_to_ee,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "oranges_cfg": [
                    SceneEntityCfg("Orange001"),
                    SceneEntityCfg("Orange002"),
                    SceneEntityCfg("Orange003"),
                ],
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # rsl_rl needs a single flat vector

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickOrangeRLRewardsCfg:
    """Reward terms for RL training."""

    # Shaped: guide EE toward the nearest orange
    ee_guidance = RewTerm(
        func=mdp.ee_to_nearest_orange_reward,
        weight=0.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "oranges_cfg": [
                SceneEntityCfg("Orange001"),
                SceneEntityCfg("Orange002"),
                SceneEntityCfg("Orange003"),
            ],
        },
    )

    # Per-orange grasp rewards
    grasped_001 = RewTerm(
        func=mdp.orange_grasped_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("Orange001"),
        },
    )
    grasped_002 = RewTerm(
        func=mdp.orange_grasped_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("Orange002"),
        },
    )
    grasped_003 = RewTerm(
        func=mdp.orange_grasped_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("Orange003"),
        },
    )

    # Per-orange placement rewards
    placed_001 = RewTerm(
        func=mdp.orange_placed_reward,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("Orange001"),
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )
    placed_002 = RewTerm(
        func=mdp.orange_placed_reward,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("Orange002"),
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )
    placed_003 = RewTerm(
        func=mdp.orange_placed_reward,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("Orange003"),
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )

    # Sparse task-completion bonus
    task_complete = RewTerm(
        func=mdp.task_complete_bonus,
        weight=10.0,
        params={
            "oranges_cfg": [
                SceneEntityCfg("Orange001"),
                SceneEntityCfg("Orange002"),
                SceneEntityCfg("Orange003"),
            ],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class PickOrangeRLEnvCfg(PickOrangeEnvCfg):
    """pick_orange environment configured for RL training with rsl_rl."""

    observations: PickOrangeRLObsCfg = PickOrangeRLObsCfg()
    rewards: PickOrangeRLRewardsCfg = PickOrangeRLRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Joint position control — no IK, gravity disabled for arm stability
        self.use_teleop_device("so101_joint_pos")

        # Shorter episode is sufficient for RL exploration
        self.episode_length_s = 30.0

        # Cameras not needed for proprioceptive RL
        self.scene.wrist = None
        self.scene.front = None

        # Disable recorder for training runs
        self.recorders = None
