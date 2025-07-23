from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Toy Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

LIGHTWHEEL_TOYROOM_USD_PATH = str(SCENES_ROOT / "lightwheel_toyroom" / "scene.usd")

LIGHTWHEEL_TOYROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIGHTWHEEL_TOYROOM_USD_PATH,
    )
)
