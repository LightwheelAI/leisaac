import torch

from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_ORANGE_USD_PATH
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization

from . import mdp


@configclass
class PickOrangeSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.5, 0.6), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True
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
class ActionsCfg:
    """Configuration for the actions."""
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class EventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False})
        front = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Configuration for the rewards"""


@configclass
class TerminationsCfg:
    """Configuration for the termination"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=mdp.task_done, params={
        "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
        "plate_cfg": SceneEntityCfg("Plate")
    })


@configclass
class PickOrangeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (2.0, -0.5, 1.0)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        parse_usd_and_create_subassets(KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=['Orange001', 'Orange002', 'Orange003', 'Plate'])

        domain_randomization(self, random_options=[
            randomize_object_uniform("Orange001", pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}),
            randomize_object_uniform("Orange002", pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}),
            randomize_object_uniform("Orange003", pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}),
            randomize_object_uniform("Plate", pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}),
            randomize_camera_uniform("front", pose_range={
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05),
                "roll": (-5 * torch.pi / 180, 5 * torch.pi / 180),
                "pitch": (-5 * torch.pi / 180, 5 * torch.pi / 180),
                "yaw": (-5 * torch.pi / 180, 5 * torch.pi / 180)}, convention="ros"),
        ])

    def use_teleop_device(self, teleop_device) -> None:
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
