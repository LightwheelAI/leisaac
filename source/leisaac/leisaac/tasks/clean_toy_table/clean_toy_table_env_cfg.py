from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.toyroom import (
    LIGHTWHEEL_TOYROOM_CFG,
    LIGHTWHEEL_TOYROOM_USD_PATH,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


@configclass
class CleanToyTableSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the clean top table task."""

    scene: AssetBaseCfg = LIGHTWHEEL_TOYROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.objs_in_box,
        params={
            "object_cfg_list": [SceneEntityCfg("Character_E"), SceneEntityCfg("Character_E_1")],
            "box_cfg": SceneEntityCfg("Box"),
        },
    )


@configclass
class CleanToyTableEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the clean top table environment."""

    scene: CleanToyTableSceneCfg = CleanToyTableSceneCfg(env_spacing=8.0)

    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-1.5, -2.0, 1.5)
        self.viewer.lookat = (-0.2, -0.3, 0.5)

        self.scene.robot.init_state.pos = (-0.42, -0.26, 0.43)

        parse_usd_and_create_subassets(LIGHTWHEEL_TOYROOM_USD_PATH, self)
