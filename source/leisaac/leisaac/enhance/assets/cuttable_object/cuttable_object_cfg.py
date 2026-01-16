"""Configuration for cuttable objects."""

from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .cuttable_object import SingleCuttableObject


@configclass
class CuttableObjectCfg(AssetBaseCfg):
    """Configuration for the cuttable object."""

    usd_path: str = ""
    """Path to the USD file for the cuttable object."""
    mesh_subfix: str = ""
    """Subfix path to the mesh prim."""
    trigger_subfix: str = "Trigger/Cube"
    """Subfix path to the trigger collision mesh."""
    knife_prim_path: str = ""
    """Full prim path to the knife mesh."""
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Base position of the object."""
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Base quaternion (wxyz) of the object."""
    class_type: type = SingleCuttableObject
    """Class type of the cuttable object."""
