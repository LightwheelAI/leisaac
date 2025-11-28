import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .. import mdp
from ..pick_orange_env_cfg import PickOrangeSceneCfg


@configclass
class PickOrangeEnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for the pick orange task."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=["Orange001", "Orange002", "Orange003", "Plate"]
        )

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "Orange001", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                ),
                randomize_object_uniform(
                    "Orange002", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                ),
                randomize_object_uniform(
                    "Orange003", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                ),
                randomize_object_uniform("Plate", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.025, 0.025),
                        "y": (-0.025, 0.025),
                        "z": (-0.025, 0.025),
                        "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )


class PickOrangeEnv(SingleArmTaskDirectEnv):
    """Direct env for the pick orange task."""

    cfg: PickOrangeEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # add subtask observation
        obs["subtask_terms"] = {
            "pick_orange001": mdp.orange_grasped(self, object_cfg=SceneEntityCfg("Orange001")),
            "put_orange001_to_plate": mdp.put_orange_to_plate(
                self, object_cfg=SceneEntityCfg("Orange001"), plate_cfg=SceneEntityCfg("Plate")
            ),
            "pick_orange002": mdp.orange_grasped(self, object_cfg=SceneEntityCfg("Orange002")),
            "put_orange002_to_plate": mdp.put_orange_to_plate(
                self, object_cfg=SceneEntityCfg("Orange002"), plate_cfg=SceneEntityCfg("Plate")
            ),
            "pick_orange003": mdp.orange_grasped(self, object_cfg=SceneEntityCfg("Orange003")),
            "put_orange003_to_plate": mdp.put_orange_to_plate(
                self, object_cfg=SceneEntityCfg("Orange003"), plate_cfg=SceneEntityCfg("Plate")
            ),
        }
        return obs

    def _check_success(self) -> torch.Tensor:
        return mdp.task_done(
            env=self,
            oranges_cfg=[SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            plate_cfg=SceneEntityCfg("Plate"),
        )
