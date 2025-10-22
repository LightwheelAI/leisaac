import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.toyroom import LIGHTWHEEL_TOYROOM_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from .. import mdp
from ..clean_toy_table_bi_arm_env_cfg import CleanToyTableBiArmSceneCfg
from ...template import BiArmTaskDirectEnvCfg, BiArmTaskDirectEnv


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

        self.state_space['left'] = [self.scene.left_wrist.height, self.scene.left_wrist.width, 3]
        self.state_space['right'] = [self.scene.right_wrist.height, self.scene.right_wrist.width, 3]
        self.state_space['top'] = [self.scene.top.height, self.scene.top.width, 3]
        self.observation_space['left'] = [self.scene.left_wrist.height, self.scene.left_wrist.width, 3]
        self.observation_space['right'] = [self.scene.right_wrist.height, self.scene.right_wrist.width, 3]
        self.observation_space['top'] = [self.scene.top.height, self.scene.top.width, 3]

        parse_usd_and_create_subassets(LIGHTWHEEL_TOYROOM_USD_PATH, self)


class CleanToyTableBiArmEnv(BiArmTaskDirectEnv):
    """Direct env for the clean toy table task."""
    cfg: CleanToyTableBiArmEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # add image observation
        obs['policy']['left'] = mdp.image(self, sensor_cfg=SceneEntityCfg("left_wrist"), data_type="rgb", normalize=False)
        obs['policy']['right'] = mdp.image(self, sensor_cfg=SceneEntityCfg("right_wrist"), data_type="rgb", normalize=False)
        obs['policy']['top'] = mdp.image(self, sensor_cfg=SceneEntityCfg("top"), data_type="rgb", normalize=False)
        return obs

    def _check_success(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
