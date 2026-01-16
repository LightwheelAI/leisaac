"""Cuttable object asset for mesh cutting simulation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import omni
import torch
from isaacsim.core.utils.prims import delete_prim
from leisaac.utils.collision_checker import Collision_Checker
from leisaac.utils.cutMeshNode import cutMeshNode
from pxr import UsdGeom

if TYPE_CHECKING:
    from .cuttable_object_cfg import CuttableObjectCfg


class SingleCuttableObject:
    """Single cuttable object that wraps cutMeshNode functionality."""

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        mesh_subfix: str,
        trigger_subfix: str,
        knife_path: str,
        base_pos: tuple[float, float, float],
        base_quat: tuple[float, float, float, float],
    ):
        self._prim_path = prim_path
        self._usd_path = usd_path
        self._mesh_path = f"{prim_path}/{mesh_subfix}" if mesh_subfix else prim_path
        self._trigger_path = f"{prim_path}/{trigger_subfix}"
        self._knife_path = knife_path
        self._base_pos = base_pos
        self._base_quat = base_quat

        self._stage = None
        self._collision_checker = None
        self._dummy_db = None
        self._last_collision = False

    def _setup_collision_checker(self):
        self._collision_checker = Collision_Checker(
            stage=self._stage,
            prim_path0=self._trigger_path,
            prim_path1=self._knife_path,
        )
        self._last_collision = False

    def _setup_cut_mesh_node(self):
        self._dummy_db.inputs.cut_mesh_path = self._mesh_path
        self._dummy_db.inputs.knife_mesh_path = self._knife_path
        self._dummy_db.internal_state = cutMeshNode.internal_state()
        self._dummy_db.inputs.cutEventIn = False

    def initialize(self):
        self._stage = omni.usd.get_context().get_stage()
        self._dummy_db = SimpleNamespace()
        self._dummy_db.inputs = SimpleNamespace()
        self._setup_cut_mesh_node()
        self._setup_collision_checker()

    def step(self):
        if_collision, _, _ = self._collision_checker.meshes_aabb_collide()
        if if_collision == self._last_collision and if_collision:
            if_collision = False
        else:
            self._last_collision = if_collision
        self._dummy_db.inputs.cutEventIn = if_collision
        cutMeshNode.compute(self._dummy_db)

    def _respawn(self, position: tuple[float, float, float], orientation: tuple[float, float, float, float]):
        """Delete and respawn the object at the given pose, then reinitialize cutting system."""
        delete_prim(self._prim_path)
        cfg = sim_utils.UsdFileCfg(usd_path=self._usd_path)
        cfg.func(self._prim_path, cfg, translation=position, orientation=orientation)
        self._setup_cut_mesh_node()
        self._setup_collision_checker()

    def reset(self):
        """Reset to base position."""
        self._respawn(self._base_pos, self._base_quat)

    def reset_to(self, position: tuple[float, float, float], orientation: tuple[float, float, float, float]):
        """Reset to specified position."""
        self._respawn(position, orientation)

    @property
    def piece_count(self) -> int:
        mesh_prim = self._stage.GetPrimAtPath(self._mesh_path)
        return len([p for p in mesh_prim.GetChildren() if p.IsActive()])

    @property
    def root_pose_w(self) -> torch.Tensor:
        prim = self._stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            return torch.tensor([*self._base_pos, *self._base_quat], dtype=torch.float32)

        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(0)
        pos = transform.ExtractTranslation()
        rot = transform.ExtractRotationQuat()
        return torch.tensor([pos[0], pos[1], pos[2], rot.GetReal(), *rot.GetImaginary()], dtype=torch.float32)


class CuttableObject:
    """Manages cuttable object instances across environments."""

    cfg: CuttableObjectCfg

    def __init__(self, cfg: CuttableObjectCfg):
        self.cfg = cfg
        self.cuttable_objects: list[SingleCuttableObject] = []

        matching_prims = sim_utils.find_matching_prim_paths(self.cfg.prim_path)
        for prim_path in matching_prims:
            self.cuttable_objects.append(
                SingleCuttableObject(
                    prim_path=prim_path,
                    usd_path=self.cfg.usd_path,
                    mesh_subfix=self.cfg.mesh_subfix,
                    trigger_subfix=self.cfg.trigger_subfix,
                    knife_path=self.cfg.knife_prim_path,
                    base_pos=self.cfg.base_pos,
                    base_quat=self.cfg.base_quat,
                )
            )

    def initialize(self):
        for obj in self.cuttable_objects:
            obj.initialize()

    def step(self):
        for obj in self.cuttable_objects:
            obj.step()

    def reset(self, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = range(len(self.cuttable_objects))
        for env_id in env_ids:
            self.cuttable_objects[env_id].reset()

    def reset_to(self, env_ids: list[int], positions: torch.Tensor, orientations: torch.Tensor):
        for i, env_id in enumerate(env_ids):
            pos = tuple(positions[i].cpu().numpy())
            ori = tuple(orientations[i].cpu().numpy())
            self.cuttable_objects[env_id].reset_to(pos, ori)

    @property
    def piece_counts(self) -> torch.Tensor:
        return torch.tensor([obj.piece_count for obj in self.cuttable_objects], dtype=torch.int32)

    def check_success(self, min_count: int = 2) -> torch.Tensor:
        return self.piece_counts >= min_count

    @property
    def root_pose_w(self) -> torch.Tensor:
        poses = [obj.root_pose_w for obj in self.cuttable_objects]
        return torch.stack(poses, dim=0)
