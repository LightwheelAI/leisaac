import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from leisaac.assets.robots.lerobot import SO101_KINFE_CFG
from leisaac.assets.scenes.kitchen import (
    KITCHEN_WITH_SAUSAGE_CFG,
    KITCHEN_WITH_SAUSAGE_USD_PATH,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    BiArmObservationsCfg,
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmTerminationsCfg,
)


@configclass
class SausageCutBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the sausage cut task using two arms."""

    # Use merged USD that includes sausage
    scene: AssetBaseCfg = KITCHEN_WITH_SAUSAGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # Replace right arm with knife-equipped robot (use standard config to match lehome)
    right_arm: ArticulationCfg = SO101_KINFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Right_Robot")


@configclass
class SausageCutBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the sausage cut environment."""

    scene: SausageCutBiArmSceneCfg = SausageCutBiArmSceneCfg(env_spacing=4.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    # Simulation configuration - CRITICAL: use_fabric=False is required for DeformableBody/cutting
    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Camera positions
        self.viewer.eye = (2.5, -5.0, 1.6)
        self.viewer.lookat = (3.7, -6.15, 0.84)

        # Robot initial positions
        self.scene.left_arm.init_state.pos = (3.4, -5.85, 0.768)
        self.scene.left_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.scene.right_arm.init_state.pos = (3.4, -6.45, 0.768)
        self.scene.right_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        # Simulation settings
        self.decimation = 2
        self.dynamic_reset_gripper_effort_limit = False

        # Parse USD for additional assets (ChoppingBlock, etc.)
        parse_usd_and_create_subassets(
            KITCHEN_WITH_SAUSAGE_USD_PATH,
            self,
            exclude_name_list=["Sausage"],  # Sausage is handled separately due to cutting
        )
