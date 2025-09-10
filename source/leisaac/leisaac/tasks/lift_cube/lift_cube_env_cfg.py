import torch

from dataclasses import MISSING
from typing import Any, Dict, List

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization
from leisaac.enhance.envs.manager_based_rl_digital_twin_env_cfg import ManagerBasedRLDigitalTwinEnvCfg

from . import mdp


@configclass
class LiftCubeSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the lift cube task."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"),  # no offset for ik convert
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/jaw", name="jaw", offset=OffsetCfg(pos=(-0.021, -0.070, 0.02)))  # set offset for cube detection
        ]
    )

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.6, -0.75, 0.38), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
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
        front = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False})
        ee_frame_state = ObsTerm(func=mdp.ee_frame_state, params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")})
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""
        pick_cube = ObsTerm(func=mdp.object_grasped, params={"robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class RewardsCfg:
    """Configuration for the rewards"""


@configclass
class TerminationsCfg:
    """Configuration for the termination"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=mdp.cube_height_above_base, params={
        "cube_cfg": SceneEntityCfg("cube"),
        "robot_cfg": SceneEntityCfg("robot"),
        "robot_base_name": "base",
        "height_threshold": 0.20
    })


@configclass
class LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lift cube environment."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)

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
        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        self.scene.ee_frame.visualizer_cfg.markers['frame'].scale = (0.05, 0.05, 0.05)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        domain_randomization(self, random_options=[
            randomize_object_uniform("cube", pose_range={
                "x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (0.0, 0.0),
                "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180)}),
            randomize_camera_uniform("front", pose_range={
                "x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005),
                "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180)}, convention="opengl"),
        ])

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)


@configclass
class LiftCubeDigitalTwinEnvCfg(LiftCubeEnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
    """Configuration for the lift cube digital twin environment."""

    rgb_overlay_mode: str = "background"

    rgb_overlay_paths: Dict[str, str] = {
        "front": "greenscreen/background-lift-cube.png"
    }

    render_objects: List[SceneEntityCfg] = [
        SceneEntityCfg("cube"),
        SceneEntityCfg("robot"),
    ]
