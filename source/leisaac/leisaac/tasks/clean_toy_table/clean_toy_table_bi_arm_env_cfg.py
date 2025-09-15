from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.toyroom import LIGHTWHEEL_TOYROOM_CFG, LIGHTWHEEL_TOYROOM_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import BiArmTaskSceneCfg, BiArmTaskEnvCfg, BiArmObservationsCfg, BiArmTerminationsCfg


@configclass
class CleanToyTableBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the clean top table task using two arms."""

    scene: AssetBaseCfg = LIGHTWHEEL_TOYROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class CleanToyTableBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the clean top table environment."""

    scene: CleanToyTableBiArmSceneCfg = CleanToyTableBiArmSceneCfg(env_spacing=8.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-1.5, -2.0, 1.5)
        self.viewer.lookat = (-0.2, -0.3, 0.5)

        self.scene.left_arm.init_state.pos = (-0.6, -0.2, 0.43)
        self.scene.right_arm.init_state.pos = (-0.15, -0.2, 0.43)

        parse_usd_and_create_subassets(LIGHTWHEEL_TOYROOM_USD_PATH, self)
