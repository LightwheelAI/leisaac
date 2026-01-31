from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

KITCHEN_WITH_ORANGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_orange" / "scene.usd")

KITCHEN_WITH_ORANGE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_ORANGE_USD_PATH,
    )
)


KITCHEN_WITH_HAMBURGER_USD_PATH = str(SCENES_ROOT / "kitchen_with_burger" / "scene.usd")
KITCHEN_WITH_HAMBURGER_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_HAMBURGER_USD_PATH,
    )
)


KITCHEN_WITH_SAUSAGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_sausage" / "scene.usd")

KITCHEN_WITH_SAUSAGE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_SAUSAGE_USD_PATH,
    )
)

# Sausage object USD path
SAUSAGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_sausage" / "objects" / "Sausage001" / "Sausage001.usd")
