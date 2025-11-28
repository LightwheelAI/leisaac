from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg as RecordTerm,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action

from . import mdp


@configclass
class BiArmTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the bi arm task."""

    scene: AssetBaseCfg = MISSING

    left_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Left_Robot")

    right_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Right_Robot")

    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.225, -0.5, 0.6), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class BiArmActionsCfg:
    """Configuration for the actions."""

    left_arm_action: mdp.ActionTermCfg = MISSING
    left_gripper_action: mdp.ActionTermCfg = MISSING
    right_arm_action: mdp.ActionTermCfg = MISSING
    right_gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class BiArmEventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class BiArmObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("left_arm")})
        left_joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("left_arm")})

        right_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("right_arm")})
        right_joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("right_arm")})

        actions = ObsTerm(func=mdp.last_action)
        left_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False}
        )
        right_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False}
        )
        top = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class BiArmRewardsCfg:
    """Configuration for the rewards"""


@configclass
class BiArmTerminationsCfg:
    """Configuration for the termination"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BiArmTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bi arm task template environment."""

    scene: BiArmTaskSceneCfg = MISSING

    observations: BiArmObservationsCfg = MISSING
    actions: BiArmActionsCfg = BiArmActionsCfg()
    events: BiArmEventCfg = BiArmEventCfg()

    rewards: BiArmRewardsCfg = BiArmRewardsCfg()
    terminations: BiArmTerminationsCfg = MISSING

    recorders: RecordTerm = RecordTerm()

    dynamic_reset_gripper_effort_limit: bool = True
    """Whether to dynamically reset the gripper effort limit."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (2.5, -1.0, 1.3)
        self.viewer.lookat = (3.6, -0.4, 1.0)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.left_arm.init_state.pos = (3.4, -0.65, 0.89)
        self.scene.right_arm.init_state.pos = (3.8, -0.65, 0.89)

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
