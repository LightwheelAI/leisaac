import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.toyroom import LIGHTWHEEL_TOYROOM_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg
from .. import mdp
from ..clean_toy_table_bi_arm_env_cfg import CleanToyTableBiArmSceneCfg


@configclass
class CleanToyTableBiArmEnvCfg(BiArmTaskDirectEnvCfg):
    """Direct env configuration for the clean toy table task."""

    scene: CleanToyTableBiArmSceneCfg = CleanToyTableBiArmSceneCfg(env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-1.5, -2.0, 1.5)
        self.viewer.lookat = (-0.2, -0.3, 0.5)

        self.scene.left_arm.init_state.pos = (-0.6, -0.2, 0.43)
        self.scene.right_arm.init_state.pos = (-0.15, -0.2, 0.43)

        parse_usd_and_create_subassets(LIGHTWHEEL_TOYROOM_USD_PATH, self)


class CleanToyTableBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for the clean toy table task."""

    cfg: CleanToyTableBiArmEnvCfg

    def _get_observations(self) -> dict:
        return super()._get_observations()

    def _check_success(self) -> torch.Tensor:
        return mdp.objs_in_box(
            env=self,
            object_cfg_list=[SceneEntityCfg("Character_E"), SceneEntityCfg("Character_E_1")],
            box_cfg=SceneEntityCfg("Box"),
        )
