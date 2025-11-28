from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.bedroom import (
    LIGHTWHEEL_BEDROOM_CFG,
    LIGHTWHEEL_BEDROOM_USD_PATH,
)
from leisaac.enhance.assets import ClothObjectCfg
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    BiArmObservationsCfg,
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmTerminationsCfg,
)


@configclass
class FoldClothBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the fold cloth task using two arms."""

    scene: AssetBaseCfg = LIGHTWHEEL_BEDROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    cloths: ClothObjectCfg = ClothObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/cloth",
        mesh_subfix="cloth/sim_cloth/sim_cloth",
        particle_system_subfix="cloth/ParticleSystem",
    )


@configclass
class FoldClothBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the fold cloth environment."""

    scene: FoldClothBiArmSceneCfg = FoldClothBiArmSceneCfg(env_spacing=8.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.9, 7.7, 4.3)
        self.viewer.lookat = (-0.9, 8.8, 3.25)

        self.scene.left_arm.init_state.pos = (-1.0, 8.35, 3.25)
        self.scene.right_arm.init_state.pos = (-0.72, 8.35, 3.25)

        # some settings for cloth simulation
        self.sim.render.antialiasing_mode = "FXAA"
        self.decimation = 2

        self.dynamic_reset_gripper_effort_limit = False

        parse_usd_and_create_subassets(LIGHTWHEEL_BEDROOM_USD_PATH, self)
