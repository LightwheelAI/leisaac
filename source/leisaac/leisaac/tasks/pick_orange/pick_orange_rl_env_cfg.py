from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .pick_orange_env_cfg import PickOrangeEnvCfg


@configclass
class PickOrangeRLObservationsCfg:
    """Flat vector observations for RL (28D total, concatenate_terms=True)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: joint_pos(6) + joint_vel(6) + ee_frame_state(7) + oranges_rel_ee(9) = 28D."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        oranges_pos_relative_to_ee = ObsTerm(
            func=mdp.oranges_pos_relative_to_ee,
            params={
                "orange_cfgs": [
                    SceneEntityCfg("Orange001"),
                    SceneEntityCfg("Orange002"),
                    SceneEntityCfg("Orange003"),
                ],
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
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

    Overrides observations with a flat 28D vector, adds reward terms,
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
