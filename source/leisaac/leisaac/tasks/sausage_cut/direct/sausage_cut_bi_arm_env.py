"""
Sausage Cut Bi-Arm Direct Environment

A simple Direct environment for the sausage cutting task.
Only overrides cutting-specific logic; robot control and observations
are handled by BiArmTaskDirectEnv base class.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import isaaclab.sim as sim_utils
import numpy as np
import omni
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.prims import delete_prim
from leisaac.assets.scenes.kitchen import KITCHEN_WITH_SAUSAGE_USD_PATH
from leisaac.utils.collision_checker import Collision_Checker
from leisaac.utils.constant import ASSETS_ROOT
from leisaac.utils.cutMeshNode import cutMeshNode
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from scipy.spatial.transform import Rotation as R

from ...template import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg
from .. import mdp
from ..sausage_cut_bi_arm_env_cfg import SausageCutBiArmSceneCfg

# Sausage USD path (for dynamic reload after cutting)
SAUSAGE_USD_PATH = str(Path(ASSETS_ROOT) / "scenes/kitchen_with_sausage/objects/Sausage001/Sausage001.usd")

# Sausage initial pose
SAUSAGE_BASE_T = (3.6612, -6.236, 0.84059)
SAUSAGE_BASE_Q_WXYZ = (-0.23287, -0.02628, 0.02471, 0.97184)


@configclass
class SausageCutBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    """Direct env configuration for the sausage cut task."""

    scene: SausageCutBiArmSceneCfg = SausageCutBiArmSceneCfg(env_spacing=4.0)

    # CRITICAL: use_fabric=False is required for mesh cutting
    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (2.5, -5.0, 1.6)
        self.viewer.lookat = (3.7, -6.15, 0.84)

        self.scene.left_arm.init_state.pos = (3.4, -5.85, 0.768)
        self.scene.left_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.scene.right_arm.init_state.pos = (3.4, -6.45, 0.768)
        self.scene.right_arm.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.decimation = 2
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
        )

        parse_usd_and_create_subassets(
            KITCHEN_WITH_SAUSAGE_USD_PATH, self, exclude_name_list=["Sausage"]  # Handled separately due to cutting
        )


class SausageCutBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for sausage cutting task."""

    cfg: SausageCutBiArmEnvCfg

    def __init__(self, cfg: SausageCutBiArmEnvCfg, render_mode: str | None = None, **kwargs):
        # Cutting system state
        self.dummy_db = SimpleNamespace(
            inputs=SimpleNamespace(cut_mesh_path=None, knife_mesh_path=None, cutEventIn=False),
            internal_state=cutMeshNode.internal_state(),
        )
        self.sausage_count = 1
        self.last_if_collision = False
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup scene and initialize cutting system."""
        super()._setup_scene()

        env_prim_path = self.scene.env_prim_paths[0]
        self.sausage_prim_path = f"{env_prim_path}/Scene/Sausage001"

        # Set cutting mesh paths
        self.dummy_db.inputs.cut_mesh_path = f"{self.sausage_prim_path}/Sausage001"
        self.dummy_db.inputs.knife_mesh_path = f"{env_prim_path}/Right_Robot/gripper/Knife/Knife/Cube"

        # Initialize collision checker
        self.stage = omni.usd.get_context().get_stage()

        # Debug: print paths and verify prims exist
        print(f"[DEBUG] env_prim_path: {env_prim_path}")
        print(f"[DEBUG] sausage_prim_path: {self.sausage_prim_path}")
        print(f"[DEBUG] cut_mesh_path: {self.dummy_db.inputs.cut_mesh_path}")
        print(f"[DEBUG] knife_mesh_path: {self.dummy_db.inputs.knife_mesh_path}")

        trigger_path = f"{self.sausage_prim_path}/Trigger/Cube"
        knife_path = f"{env_prim_path}/Right_Robot/gripper/Knife/Knife/Knife002"
        print(f"[DEBUG] trigger_path: {trigger_path}")
        print(f"[DEBUG] knife_collision_path: {knife_path}")

        # Verify prims exist
        trigger_prim = self.stage.GetPrimAtPath(trigger_path)
        knife_prim = self.stage.GetPrimAtPath(knife_path)
        print(f"[DEBUG] trigger_prim exists: {trigger_prim.IsValid() if trigger_prim else False}")
        print(f"[DEBUG] knife_prim exists: {knife_prim.IsValid() if knife_prim else False}")

        self.collision_checker = Collision_Checker(
            stage=self.stage,
            prim_path0=trigger_path,
            prim_path1=knife_path,
        )

        # Spawn sausage
        if os.path.exists(SAUSAGE_USD_PATH):
            cfg = sim_utils.UsdFileCfg(usd_path=SAUSAGE_USD_PATH)
            cfg.func(self.sausage_prim_path, cfg, translation=SAUSAGE_BASE_T, orientation=SAUSAGE_BASE_Q_WXYZ)
            print(f"[DEBUG] Sausage spawned at {self.sausage_prim_path}")
        else:
            print(f"[ERROR] Sausage USD not found: {SAUSAGE_USD_PATH}")

    def _apply_action(self) -> None:
        """Apply robot actions and handle cutting logic."""
        super()._apply_action()

        # Collision detection with debounce (protected from mesh modification errors)
        try:
            if_collision, _, _ = self.collision_checker.meshes_aabb_collide()
            if if_collision and if_collision == self.last_if_collision:
                if_collision = False
            self.last_if_collision = if_collision
        except Exception as e:
            # Mesh may be invalid during/after cutting, skip collision check
            print(f"[DEBUG] Collision check exception: {e}")
            if_collision = False

        # Trigger mesh cutting
        self.dummy_db.inputs.cutEventIn = if_collision
        try:
            cutMeshNode.compute(self.dummy_db)
        except Exception as e:
            print(f"[ERROR] cutMeshNode.compute failed: {e}")
            import traceback

            traceback.print_exc()

        if if_collision:
            self.sausage_count = self._count_sausage_meshes()
            print(f"[DEBUG] Collision detected, sausage_count: {self.sausage_count}")

    def _check_success(self) -> torch.Tensor:
        """Check if sausage has been cut."""
        return mdp.sausage_cut(env=self, min_sausage_count=2, check_rest_pose=False)

    def _count_sausage_meshes(self) -> int:
        """Count active sausage mesh pieces."""
        try:
            prim = self.stage.GetPrimAtPath(self.sausage_prim_path)
            if prim and prim.IsValid():
                return len([p for p in prim.GetChildren() if p.IsActive()])
        except Exception:
            pass
        return 1

    def _reset_sausage(self):
        """Reset sausage by deleting and recreating with randomized pose."""
        delete_prim(self.sausage_prim_path)

        # Randomize pose (translation: +/-4cm, rotation: +/-20deg on z-axis)
        rng = np.random.default_rng()
        t_new = np.array(SAUSAGE_BASE_T) + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), 0.0])

        q_wxyz = np.array(SAUSAGE_BASE_Q_WXYZ)
        r_base = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        r_new = r_base * R.from_rotvec(np.deg2rad(rng.uniform(-20, 20)) * np.array([0, 0, 1]))
        q_new = r_new.as_quat()
        q_new_wxyz = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])

        # Recreate sausage
        if os.path.exists(SAUSAGE_USD_PATH):
            cfg = sim_utils.UsdFileCfg(usd_path=SAUSAGE_USD_PATH)
            cfg.func(self.sausage_prim_path, cfg, translation=t_new, orientation=q_new_wxyz)

        # Reset cutting state
        self.dummy_db.inputs.cut_mesh_path = f"{self.sausage_prim_path}/Sausage001"
        self.dummy_db.internal_state = cutMeshNode.internal_state()
        self.dummy_db.inputs.cutEventIn = False
        self.sausage_count = 1
        self.last_if_collision = False

        # Reinitialize collision checker (prim was deleted and recreated)
        env_prim_path = self.scene.env_prim_paths[0]
        self.collision_checker = Collision_Checker(
            stage=self.stage,
            prim_path0=f"{self.sausage_prim_path}/Trigger/Cube",
            prim_path1=f"{env_prim_path}/Right_Robot/gripper/Knife/Knife/Knife002",
        )

    def _reset_idx(self, env_ids):
        """Reset environment."""
        super()._reset_idx(env_ids)
        self._reset_sausage()
