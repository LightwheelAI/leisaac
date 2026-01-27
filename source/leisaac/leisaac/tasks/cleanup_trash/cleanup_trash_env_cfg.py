from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.loft import LOFT_CFG, LOFT_USD_PATH
from leisaac.enhance.envs import mdp
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    LeKiwiObservationsCfg,
    LeKiwiTaskEnvCfg,
    LeKiwiTaskSceneCfg,
    LeKiwiTerminationsCfg,
)


@configclass
class CleanupTrashSceneCfg(LeKiwiTaskSceneCfg):
    """Scene configuration for the cleanup trash task."""

    scene: AssetBaseCfg = LOFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class TerminationsCfg(LeKiwiTerminationsCfg):

    success = DoneTerm(
        func=mdp.object_in_container,
        params={
            "object_cfg": SceneEntityCfg("Tissue005"),
            "container_cfg": SceneEntityCfg("GarbageCan003_1"),
            "x_range": (-0.10, 0.10),
            "y_range": (-0.10, 0.10),
            "height_threshold": 0.05,
        },
    )


@configclass
class CleanupTrashEnvCfg(LeKiwiTaskEnvCfg):
    """Configuration for the cleanup trash environment."""

    scene: CleanupTrashSceneCfg = CleanupTrashSceneCfg(env_spacing=8.0)

    observations: LeKiwiObservationsCfg = LeKiwiObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Pick up tissue trash from the floor and throw it into the trash bin."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-1.4, 0.4, 1.2)
        self.viewer.lookat = (-4.4, 3.6, -2.7)

        self.scene.robot.init_state.pos = (-2.3, 0.8, 0.035)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        parse_usd_and_create_subassets(LOFT_USD_PATH, self, specific_name_list=["GarbageCan003", "Tissue005"])

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "Tissue005",
                    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "GarbageCan003_1",
                    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                ),
            ],
        )
