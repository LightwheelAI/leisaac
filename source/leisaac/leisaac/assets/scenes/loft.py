from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Loft Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

LOFT_USD_PATH = str(SCENES_ROOT / "lightwheel_loft" / "LW_Loft.usd")

LOFT_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LOFT_USD_PATH,
    )
)
