"""
Assemble Hamburger Bi-Arm Scene Configuration (LeIsaac Pattern)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import (
    KITCHEN_WITH_HAMBURGER_CFG,
    KITCHEN_WITH_HAMBURGER_USD_PATH,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_deformable_object_uniform,
    randomize_mixed_objects_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    BiArmObservationsCfg,
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmTerminationsCfg,
)
from . import mdp


@configclass
class AssembleHamburgerBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration - using leisaac automated loading pattern."""

    scene: AssetBaseCfg = KITCHEN_WITH_HAMBURGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class TerminationsCfg(BiArmTerminationsCfg):
    """Termination configuration for the assemble hamburger task."""

    success = DoneTerm(
        func=mdp.burger_assembly_success,
        params={
            "beef_cfg": SceneEntityCfg("Burger_Beef_Patties001"),
            "plate_cfg": SceneEntityCfg("Burger_Plate"),
        },
    )


@configclass
class AssembleHamburgerBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the assemble hamburger environment (Manager-Based)."""

    scene: AssembleHamburgerBiArmSceneCfg = AssembleHamburgerBiArmSceneCfg(env_spacing=4.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Pick the beef patties and place it on the plate"

    # Simulation configuration
    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1, render=render_cfg, use_fabric=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Aligned with leisaac kitchen reference system
        self.viewer.eye = (2.4, -5.577, 1.52)
        self.viewer.lookat = (4.03, -6.41, 0.72)

        # Robot positions - Aligned for burger kitchen
        self.scene.left_arm.init_state.pos = (3.4, -5.8, 0.78)
        self.scene.left_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.scene.right_arm.init_state.pos = (3.4, -6.4, 0.78)
        self.scene.right_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.decimation = 1
        self.dynamic_reset_gripper_effort_limit = False

        # Add lighting
        self.scene.light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
        )

        # Automatically parse and add burger components to the scene
        parse_usd_and_create_subassets(KITCHEN_WITH_HAMBURGER_USD_PATH, self)

        # Domain randomization - plate, bread, cheese bound together (lehome style)
        domain_randomization(
            self,
            random_options=[
                randomize_mixed_objects_uniform(
                    rigid_names=["Burger_Plate", "Burger_Bread002"],
                    deformable_names=["Burger_Cheese001"],
                    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                ),
                randomize_deformable_object_uniform(
                    "Burger_Beef_Patties001",
                    pose_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
                ),
            ],
        )
