from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .pick_orange_env_cfg import PickOrangeEnvCfg

_ORANGE_CFGS = [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")]
_PLATE_CFG = SceneEntityCfg("Plate")


@configclass
class PickOrangeRLObservationsCfg:
    """Flat vector observations for RL (37D total, concatenate_terms=True).

    joint_pos(6) + joint_vel(6) + ee_frame_state(7) +
    oranges_rel_ee(9) + plate_rel_ee(3) + task_status(3) + gripper_state(1) = 35D
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        # Position of each orange relative to EE (9D): tells policy where to reach
        oranges_pos_relative_to_ee = ObsTerm(
            func=mdp.oranges_pos_relative_to_ee,
            params={"orange_cfgs": _ORANGE_CFGS, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        # Position of plate relative to EE (3D): tells policy where to deposit
        plate_pos_relative_to_ee = ObsTerm(
            func=mdp.plate_pos_relative_to_ee,
            params={"plate_cfg": _PLATE_CFG, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        # Which oranges are already on the plate (3D binary): task progress signal
        oranges_task_status = ObsTerm(
            func=mdp.oranges_task_status,
            params={"orange_cfgs": _ORANGE_CFGS, "plate_cfg": _PLATE_CFG},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickOrangeRLRewardsCfg:
    """Reward terms for pick-orange RL training."""

    ee_guidance = RewTerm(
        func=mdp.ee_to_nearest_orange_reward,
        weight=0.5,
        params={
            "orange_cfgs": [
                SceneEntityCfg("Orange001"),
                SceneEntityCfg("Orange002"),
                SceneEntityCfg("Orange003"),
            ],
        },
    )
    grasped_001 = RewTerm(
        func=mdp.orange_grasped_reward,
        weight=1.0,
        params={"orange_cfg": SceneEntityCfg("Orange001")},
    )
    grasped_002 = RewTerm(
        func=mdp.orange_grasped_reward,
        weight=1.0,
        params={"orange_cfg": SceneEntityCfg("Orange002")},
    )
    grasped_003 = RewTerm(
        func=mdp.orange_grasped_reward,
        weight=1.0,
        params={"orange_cfg": SceneEntityCfg("Orange003")},
    )
    placed_001 = RewTerm(
        func=mdp.orange_placed_reward,
        weight=2.0,
        params={"orange_cfg": SceneEntityCfg("Orange001"), "plate_cfg": SceneEntityCfg("Plate")},
    )
    placed_002 = RewTerm(
        func=mdp.orange_placed_reward,
        weight=2.0,
        params={"orange_cfg": SceneEntityCfg("Orange002"), "plate_cfg": SceneEntityCfg("Plate")},
    )
    placed_003 = RewTerm(
        func=mdp.orange_placed_reward,
        weight=2.0,
        params={"orange_cfg": SceneEntityCfg("Orange003"), "plate_cfg": SceneEntityCfg("Plate")},
    )
    task_bonus = RewTerm(
        func=mdp.oranges_on_plate_bonus,
        weight=1.0,
        params={
            "orange_cfgs": [
                SceneEntityCfg("Orange001"),
                SceneEntityCfg("Orange002"),
                SceneEntityCfg("Orange003"),
            ],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class PickOrangeRLTerminationsCfg:
    """Terminations for RL: timeout only (success is rewarded but not terminates early)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PickOrangeRLEnvCfg(PickOrangeEnvCfg):
    """RL-specific configuration for the pick orange environment.

    Overrides observations with a flat 35D vector, adds reward terms,
    disables cameras, and configures so101ik action space.
    """

    observations: PickOrangeRLObservationsCfg = PickOrangeRLObservationsCfg()
    rewards: PickOrangeRLRewardsCfg = PickOrangeRLRewardsCfg()
    terminations: PickOrangeRLTerminationsCfg = PickOrangeRLTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Use IK-based action space (same as state machine)
        self.use_teleop_device("so101ik")

        # Disable cameras to speed up RL training
        self.scene.wrist = None
        self.scene.front = None

        # Longer episodes for RL
        self.episode_length_s = 30.0
