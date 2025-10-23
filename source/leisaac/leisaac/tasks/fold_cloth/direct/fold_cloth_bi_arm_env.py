import torch

from isaaclab.utils import configclass

from leisaac.assets.scenes.bedroom import LIGHTWHEEL_BEDROOM_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..fold_cloth_bi_arm_env_cfg import FoldClothBiArmSceneCfg
from ...template import BiArmTaskDirectEnvCfg, BiArmTaskDirectEnv


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

        parse_usd_and_create_subassets(LIGHTWHEEL_BEDROOM_USD_PATH, self)


class FoldClothBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for the fold cloth task."""
    cfg: FoldClothBiArmEnvCfg

    def _get_observations(self) -> dict:
        return super()._get_observations()

    def _check_success(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
