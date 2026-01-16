"""Sausage cutting task with bi-arm robot (Direct environment)."""

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from leisaac.assets.robots.lerobot import SO101_KINFE_CFG
from leisaac.assets.scenes.kitchen import (
    KITCHEN_WITH_SAUSAGE_CFG,
    KITCHEN_WITH_SAUSAGE_USD_PATH,
    SAUSAGE_USD_PATH,
)
from leisaac.enhance.assets import CuttableObject, CuttableObjectCfg
from leisaac.enhance.envs.mdp.recorders.recorders_cfg import (
    DirectEnvActionStateWithCuttableObjectsRecorderManagerCfg as CuttableObjectRecordTerm,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_cuttable_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg, BiArmTaskSceneCfg

SAUSAGE_BASE_POS = (3.6612, -6.236, 0.84059)
SAUSAGE_BASE_QUAT = (-0.23287, -0.02628, 0.02471, 0.97184)


@configclass
class SausageCutBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the sausage cut task using two arms."""

    scene: AssetBaseCfg = KITCHEN_WITH_SAUSAGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    right_arm: ArticulationCfg = SO101_KINFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Right_Robot")


@configclass
class SausageCutBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    """Direct env configuration for the sausage cut task."""

    scene: SausageCutBiArmSceneCfg = SausageCutBiArmSceneCfg(env_spacing=4.0)
    task_description: str = "Cut the sausage into pieces."

    # Render configuration with antialiasing enabled
    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1, render=render_cfg, use_fabric=False)

    # Disable IsaacLab UI window
    ui_window_class_type: type | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (2.15, -6.5, 1.9)
        self.viewer.lookat = (3.90, -6.43, 0.92)

        self.scene.left_arm.init_state.pos = (3.4, -5.85, 0.768)
        self.scene.left_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.right_arm.init_state.pos = (3.4, -6.45, 0.768)
        self.scene.right_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        # Simulation settings for deformable body cutting (already set in sim config above)
        self.decimation = 1
        self.dynamic_reset_gripper_effort_limit = False

        # Recorder for cuttable objects
        self.recorders = CuttableObjectRecordTerm()

        self.scene.light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
        )
        parse_usd_and_create_subassets(KITCHEN_WITH_SAUSAGE_USD_PATH, self, exclude_name_list=["Sausage"])

        # Domain randomization for the cuttable sausage
        domain_randomization(
            self,
            random_options=[
                randomize_cuttable_object_uniform(
                    "cuttable_sausage",
                    pose_range={
                        "x": (-0.04, 0.04),
                        "y": (-0.04, 0.04),
                        "yaw": (-20 * torch.pi / 180, 20 * torch.pi / 180),
                    },
                ),
            ],
        )


class SausageCutBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for the sausage cut task."""

    cfg: SausageCutBiArmEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        env_path = self.scene.env_prim_paths[0]
        sausage_prim_path = f"{env_path}/Scene/Sausage001"

        # Spawn sausage
        cfg = sim_utils.UsdFileCfg(usd_path=SAUSAGE_USD_PATH)
        cfg.func(sausage_prim_path, cfg, translation=SAUSAGE_BASE_POS, orientation=SAUSAGE_BASE_QUAT)

        # Initialize cuttable object
        self.cuttable_sausage = CuttableObject(
            CuttableObjectCfg(
                prim_path=sausage_prim_path,
                usd_path=SAUSAGE_USD_PATH,
                mesh_subfix="Sausage001",
                trigger_subfix="Trigger/Cube",
                knife_prim_path=f"{env_path}/Right_Robot/gripper/Knife/Knife/Knife002",
                base_pos=SAUSAGE_BASE_POS,
                base_quat=SAUSAGE_BASE_QUAT,
            )
        )

    def initialize(self):
        self.cuttable_sausage.initialize()

    def _apply_action(self) -> None:
        super()._apply_action()
        self.cuttable_sausage.step()

    def _check_success(self) -> torch.Tensor:
        return self.cuttable_sausage.check_success(min_count=2)

    def _reset_idx(self, env_ids: Sequence[int]):
        self.cuttable_sausage.reset(list(env_ids))
        super()._reset_idx(env_ids)

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        # Call super first (which calls _reset_idx and resets sausage to default position)
        super().reset_to(state, env_ids, seed, is_relative)
        # Then override with the position from HDF5 state if available
        if "cuttable_object" in state:
            for attr_name, asset_state in state["cuttable_object"].items():
                cuttable_object = getattr(self, attr_name, None)
                if cuttable_object is not None:
                    root_pose = asset_state["root_pose"].clone()
                    if is_relative:
                        root_pose[:, :3] += self.scene.env_origins[env_ids]
                    cuttable_object.reset_to(list(env_ids), root_pose[:, :3], root_pose[:, 3:])
