from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .lift_cube_env_cfg import LiftCubeEnvCfg, LiftCubeSceneCfg

_CUBE_CFG = SceneEntityCfg("cube")
_ROBOT_CFG = SceneEntityCfg("robot")


@configclass
class LiftCubeRLSceneCfg(LiftCubeSceneCfg):
    """RL-specific scene: adds jaw ContactSensor for accurate grasp detection."""

    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/cube"],
    )


@configclass
class LiftCubeRLObservationsCfg:
    """Flat vector observations for RL (22D total).

    joint_pos(6) + joint_vel(6) + ee_frame_state(7) + cube_rel_ee(3) = 22D
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": _ROBOT_CFG},
        )
        # Cube position relative to EE jaw (3D): tells policy where to reach
        cube_pos_relative_to_ee = ObsTerm(
            func=mdp.cube_pos_relative_to_ee,
            params={"cube_cfg": _CUBE_CFG, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LiftCubeRLRewardsCfg:
    """Reward terms for lift-cube RL training."""

    # Success: large one-time bonus when cube reaches target height (episode ends immediately after)
    cube_success = RewTerm(
        func=mdp.cube_success_bonus,
        weight=200.0,
        params={"cube_cfg": _CUBE_CFG, "robot_cfg": _ROBOT_CFG, "height_threshold": 0.20},
    )
    # Stage 1: tanh reaching reward [0, 1] — always has gradient toward cube
    ee_to_cube = RewTerm(
        func=mdp.ee_to_cube_reward,
        weight=1.0,
        params={"cube_cfg": _CUBE_CFG, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # Stage 2: grasped bonus — jaw contact with cube + gripper closed
    cube_grasped = RewTerm(
        func=mdp.cube_grasped_reward,
        weight=1.0,
        params={"contact_cfg": SceneEntityCfg("jaw_contact"), "robot_cfg": _ROBOT_CFG},
    )
    # Stage 3: height reward [0, 1], only when grasped — peaks at 20cm
    cube_height = RewTerm(
        func=mdp.cube_height_if_grasped,
        weight=1.0,
        params={
            "cube_cfg": _CUBE_CFG,
            "robot_cfg": _ROBOT_CFG,
            "contact_cfg": SceneEntityCfg("jaw_contact"),
        },
    )


@configclass
class LiftCubeRLTerminationsCfg:
    """Terminations: timeout + early success for efficiency."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={"cube_cfg": _CUBE_CFG, "robot_cfg": _ROBOT_CFG, "height_threshold": 0.20},
    )


@configclass
class LiftCubeRLEnvCfg(LiftCubeEnvCfg):
    """RL-specific configuration for the lift-cube environment.

    Observations: 22D flat vector.
    Action space: 7D (rl_so101leader: 6D delta EE pose + 1D binary gripper).
    Cameras disabled for faster training.
    """

    scene: LiftCubeRLSceneCfg = LiftCubeRLSceneCfg(env_spacing=8.0)
    observations: LiftCubeRLObservationsCfg = LiftCubeRLObservationsCfg()
    rewards: LiftCubeRLRewardsCfg = LiftCubeRLRewardsCfg()
    terminations: LiftCubeRLTerminationsCfg = LiftCubeRLTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.use_teleop_device("rl_so101leader")

        # Disable camera for faster RL training
        self.scene.front = None

        # Remove camera randomization events that reference the disabled front camera
        for attr_name in list(vars(self.events).keys()):
            term = getattr(self.events, attr_name, None)
            if (
                hasattr(term, "params")
                and isinstance(term.params.get("asset_cfg"), SceneEntityCfg)
                and term.params["asset_cfg"].name == "front"
            ):
                delattr(self.events, attr_name)

        self.episode_length_s = 15.0
