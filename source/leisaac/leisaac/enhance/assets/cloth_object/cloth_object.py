from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.scene import InteractiveScene
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
from isaacsim.core.simulation_manager import SimulationManager

if TYPE_CHECKING:
    from .cloth_object_cfg import ClothObjectCfg


class SingleClothObject(SingleClothPrim):
    """
    SingleClothObject class that wraps the Isaac Sim SingleCloth prim functionality.
    """

    def __init__(self, prim_path: str, mesh_subfix: str = "mesh", particle_system_subfix: str = "ParticleSystem"):
        """Single Cloth Object asset class."""
        super().__init__(
            prim_path=f"{prim_path}/{mesh_subfix}",
            particle_system=SingleParticleSystem(f"{prim_path}/{particle_system_subfix}"),
        )

    def initialize(self):
        """
        Initialize the object by setting its initial position and orientation,
        while also get initial info of particles that make up the object.
        """
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        self._cloth_prim_view.initialize(self.physics_sim_view)

        # get initial info of particles that make up the object
        self.initial_point_positions = self._cloth_prim_view.get_world_positions()
        self.init_world_pos, self.init_world_quat = self.get_world_pose()

    def reset(self):
        """
        Reset the particles points and world pose.
        """
        self._cloth_prim_view.set_world_positions(self.initial_point_positions)
        self.set_world_pose(self.init_world_pos, self.init_world_quat)

    @property
    def point_positions(self):
        return self._cloth_prim_view.get_world_positions()


class ClothObject:
    """
    Manages all single cloth object instances in the environment.
    """

    cfg: ClothObjectCfg
    """Configuration instance for the cloth object."""

    def __init__(self, cfg: ClothObjectCfg, scene: InteractiveScene):
        self.cfg = cfg

        self.cloth_objects: list[SingleClothObject] = []
        matching_prims = sim_utils.find_matching_prim_paths(self.cfg.prim_path)
        for prim_path in matching_prims:
            self.cloth_objects.append(
                self.cfg.class_type(prim_path, self.cfg.mesh_subfix, self.cfg.particle_system_subfix)
            )

    def initialize(self):
        for cloth_object in self.cloth_objects:
            cloth_object.initialize()

    def reset(self):
        for cloth_object in self.cloth_objects:
            cloth_object.reset()

    def set_world_poses(self, positions, quats):
        for i in range(len(self.cloth_objects)):
            self.cloth_objects[i].set_world_pose(positions[i], quats[i])

    def get_world_poses(self):
        pos_w_list, quat_w_list = [], []
        for cloth_object in self.cloth_objects:
            pos_w, quat_w = cloth_object.get_world_pose()  # xyz, wxyz
            pos_w_list.append(pos_w)
            quat_w_list.append(quat_w)
        return torch.stack(pos_w_list), torch.stack(quat_w_list)

    @property
    def point_positions(self):
        point_positions = []
        for cloth_object in self.cloth_objects:
            point_positions.append(cloth_object.point_positions)
        return torch.cat(point_positions, dim=0)

    @property
    def root_pose_w(self):
        pos_w, quat_w = self.get_world_poses()
        return torch.cat([pos_w, quat_w], dim=1)
