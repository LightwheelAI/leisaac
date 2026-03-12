import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .lift_cube_env_cfg import LiftCubeEnvCfg, LiftCubeSceneCfg

TRAIN_CFG = {
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.3,
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.1,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-4,
        "schedule": "adaptive",
        "gamma": 0.95,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 0.5,
    },
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 100,
    "save_interval": 50,
    "experiment_name": "lift_cube_rl",
    "seed": 42,
}

_CUBE_CFG = SceneEntityCfg("cube")
_ROBOT_CFG = SceneEntityCfg("robot")


@configclass
class LiftCubeRLObservationsCfg:
    """Flat vector observations for RL (26D total).

    joint_pos(6) + joint_vel(6) + ee_frame_state(7) + cube_rel_ee(3) + cube_quat(4) = 26D
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": _ROBOT_CFG},
        )
        cube_pos_relative_to_ee = ObsTerm(
            func=mdp.cube_pos_relative_to_ee,
            params={"cube_cfg": _CUBE_CFG, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        cube_quat = ObsTerm(
            func=mdp.cube_quat,
            params={"cube_cfg": _CUBE_CFG},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LiftCubeRLRewardsCfg:
    """Reward terms for lift-cube RL training."""

    cube_success = RewTerm(
        func=mdp.cube_success_bonus,
        weight=200.0,
        params={"cube_cfg": _CUBE_CFG, "robot_cfg": _ROBOT_CFG, "height_threshold": 0.20},
    )
    ee_to_cube = RewTerm(
        func=mdp.ee_to_cube_reward,
        weight=1.5,
        params={"cube_cfg": _CUBE_CFG, "robot_cfg": _ROBOT_CFG},
    )
    cube_height = RewTerm(
        func=mdp.cube_height_reward,
        weight=10.0,
        params={"cube_cfg": _CUBE_CFG, "robot_cfg": _ROBOT_CFG, "min_height": 0.046, "k": 3.0},
    )


@configclass
class LiftCubeRLTerminationsCfg:
    """Terminations: timeout + early success."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={"cube_cfg": _CUBE_CFG, "robot_cfg": _ROBOT_CFG, "height_threshold": 0.20},
    )


@configclass
class LiftCubeRLEnvCfg(LiftCubeEnvCfg):
    """RL-specific configuration for the lift-cube environment.

    Observations: 26D flat vector.
    Action space: 7D (rl_so101leader: 6D delta EE pose + 1D binary gripper).
    Cameras disabled for faster training.
    """

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)
    observations: LiftCubeRLObservationsCfg = LiftCubeRLObservationsCfg()
    rewards: LiftCubeRLRewardsCfg = LiftCubeRLRewardsCfg()
    terminations: LiftCubeRLTerminationsCfg = LiftCubeRLTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.use_teleop_device("rl_so101leader")

        self.scene.front = None

        for attr_name in list(vars(self.events).keys()):
            term = getattr(self.events, attr_name, None)
            if (
                hasattr(term, "params")
                and isinstance(term.params.get("asset_cfg"), SceneEntityCfg)
                and term.params["asset_cfg"].name == "front"
            ):
                delattr(self.events, attr_name)

        self.episode_length_s = 15.0

        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.damping = 8.0

        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan": 0.0,
            "shoulder_lift": math.radians(-100.0),
            "elbow_flex": math.radians(90.0),
            "wrist_flex": math.radians(50.0),
            "wrist_roll": 0.0,
            "gripper": 1.0,
        }
