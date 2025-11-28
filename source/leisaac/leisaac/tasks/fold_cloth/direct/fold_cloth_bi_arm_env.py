from collections.abc import Sequence

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.bedroom import LIGHTWHEEL_BEDROOM_USD_PATH
from leisaac.enhance.assets import ClothObject
from leisaac.enhance.envs.mdp.recorders.recorders_cfg import (
    DirectEnvActionStateWithParticleObjectsRecorderManagerCfg as ParticleObjectRecordTerm,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_particle_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg
from .. import mdp
from ..fold_cloth_bi_arm_env_cfg import FoldClothBiArmSceneCfg


@configclass
class FoldClothBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    """Direct env configuration for the fold cloth task."""

    scene: FoldClothBiArmSceneCfg = FoldClothBiArmSceneCfg(env_spacing=8.0)

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

        # recorder for particle objects enhancement
        self.recorders = ParticleObjectRecordTerm()

        parse_usd_and_create_subassets(LIGHTWHEEL_BEDROOM_USD_PATH, self)

        # domain randomization for the cloth object.
        # Note: Only works on DirectTaskEnv, as the ClothObject is not supported in ManagerBasedTaskEnv
        domain_randomization(
            self,
            random_options=[
                randomize_particle_object_uniform(
                    "cloths",
                    pose_range={
                        "yaw": (-5 * torch.pi / 180, 5 * torch.pi / 180),
                    },
                ),
            ],
        )


class FoldClothBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for the fold cloth task."""

    cfg: FoldClothBiArmEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        assert self.device != "cpu", (
            "CUDA device is recommended for FoldCloth task as cloth deformation will not work properly on CPU. You are"
            f" currently using the {self.device}."
        )
        if hasattr(self.cfg.scene, "cloths"):
            self.scene.particle_objects = (
                {} if not hasattr(self.scene, "particle_objects") else self.scene.particle_objects
            )
            self.scene.particle_objects["cloths"] = ClothObject(
                cfg=self.cfg.scene.cloths,
                scene=self.scene,
            )

    def initialize(self):
        self.scene.particle_objects["cloths"].initialize()

    def _get_observations(self) -> dict:
        return super()._get_observations()

    def _check_success(self) -> torch.Tensor:
        return mdp.cloth_folded(
            env=self,
            cloth_cfg=SceneEntityCfg("cloths"),
            cloth_keypoints_index=[159789, 120788, 115370, 159716, 121443, 112382],
            distance_threshold=0.20,
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        self.scene.particle_objects["cloths"].reset()
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
        for asset_name, particle_object in self.scene.particle_objects.items():
            asset_state = state["particle_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.scene.env_origins[env_ids]
            particle_object.set_world_poses(root_pose[:, :3], root_pose[:, 3:])
        super().reset_to(state, env_ids, seed, is_relative)
