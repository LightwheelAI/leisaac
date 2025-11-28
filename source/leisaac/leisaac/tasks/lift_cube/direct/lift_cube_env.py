import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .. import mdp
from ..lift_cube_env_cfg import LiftCubeSceneCfg


@configclass
class LiftCubeEnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for the lift cube task."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.075, 0.075),
                        "y": (-0.075, 0.075),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="opengl",
                ),
            ],
        )


class LiftCubeEnv(SingleArmTaskDirectEnv):
    """Direct env for the lift cube task."""

    cfg: LiftCubeEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # add subtask observation
        obs["subtask_terms"] = {
            "pick_cube": mdp.object_grasped(
                self,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg("cube"),
            )
        }
        return obs

    def _check_success(self) -> torch.Tensor:
        return mdp.cube_height_above_base(
            env=self,
            cube_cfg=SceneEntityCfg("cube"),
            robot_cfg=SceneEntityCfg("robot"),
            robot_base_name="base",
            height_threshold=0.20,
        )
