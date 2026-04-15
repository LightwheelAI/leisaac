from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData
from leisaac.assets.robots.elfin_s30 import S30_CFG
from leisaac.assets.scenes.kitchen import (
    KITCHEN_WITH_ORANGE_CFG,
    KITCHEN_WITH_ORANGE_USD_PATH,
)
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.constant import S30_JOINT_NAMES
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.robot_utils import convert_s30_action_to_lerobot

from ..template import SingleArmEventCfg, SingleArmObservationsCfg, SingleArmTaskEnvCfg, SingleArmTerminationsCfg
from . import mdp as s30_mdp


@configclass
class S30TaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the S30 single-arm task."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    robot: ArticulationCfg = S30_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frame tracker.
    # elfin_link6 is the last articulation link (rigid body) of the arm.
    # The gripper prim is an Xform container and NOT a rigid body, so it cannot
    # be used as a FrameTransformer target.  We attach both frames to elfin_link6
    # and apply offsets to approximate the gripper centre and finger tip.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elfin_base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/elfin_link6",
                name="gripper",
                # Offset to Robotiq base (~0.14 m along link6 z-axis after flange)
                offset=OffsetCfg(pos=(0.0, 0.0, 0.14)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/elfin_link6",
                name="jaw",
                # Offset to finger tip (~0.30 m from link6 origin)
                offset=OffsetCfg(pos=(0.0, 0.0, 0.30)),
            ),
        ],
    )

    # Wrist camera mounted on elfin_link6 (last arm link before gripper)
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elfin_link6/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(0.0, 0.7071068, 0.7071068, 0.0),  # looking forward (wxyz)
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # External front camera fixed in the scene
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elfin_base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class S30EventCfg(SingleArmEventCfg):
    """Events for S30: extends base events with Robotiq mimic-joint tracking.

    track_mimic runs every physics step (interval, step=1) to drive mimic joints
    proportional to finger_joint, replacing the ActionGraph that was removed.
    """

    track_mimic = EventTerm(
        func=s30_mdp.track_robotiq_mimic_joints,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every step
    )


@configclass
class S30ObservationsCfg(SingleArmObservationsCfg):
    pass


@configclass
class S30TerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=s30_mdp.task_done,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class S30PickOrangeEnvCfg(SingleArmTaskEnvCfg):
    """Pick-orange task environment for Elfin S30 + Robotiq 2F-140."""

    scene: S30TaskSceneCfg = S30TaskSceneCfg(env_spacing=8.0)
    observations: S30ObservationsCfg = S30ObservationsCfg()
    events: S30EventCfg = S30EventCfg()
    terminations: S30TerminationsCfg = S30TerminationsCfg()

    task_description: str = "Pick three oranges and put them into the plate, then reset the arm to rest state."

    def __post_init__(self) -> None:
        super().__post_init__()

        # Override robot name and joint feature names for S30
        self.robot_name = "elfin_s30"
        self.default_feature_joint_names = [f"{j}.pos" for j in S30_JOINT_NAMES]

        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH,
            self,
            specific_name_list=["Orange001", "Orange002", "Orange003", "Plate"],
        )

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform("Orange001", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
                randomize_object_uniform("Orange002", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
                randomize_object_uniform("Orange003", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
                randomize_object_uniform("Plate", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.025, 0.025),
                        "y": (-0.025, 0.025),
                        "z": (-0.025, 0.025),
                        "roll":  (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "yaw":   (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = episode_data._data["actions"][-1]

        if dataset_cfg.action_align:
            processed_action = convert_s30_action_to_lerobot(action.unsqueeze(0)).squeeze(0)
        else:
            processed_action = action.cpu().numpy()

        frame = {
            "action": processed_action,
            "observation.state": convert_s30_action_to_lerobot(obs_data["joint_pos"][-1].unsqueeze(0)).squeeze(0),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame
