from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Loft Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

# Base loft room scene (for sausage cutting - no sausage included)
LIGHTWHEEL_LOFTROOM_USD_PATH = str(SCENES_ROOT / "loftroom" / "scene.usd")

LIGHTWHEEL_LOFTROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIGHTWHEEL_LOFTROOM_USD_PATH,
    )
)
