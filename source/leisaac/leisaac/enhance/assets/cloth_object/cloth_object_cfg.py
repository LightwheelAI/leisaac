from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .cloth_object import SingleClothObject


@configclass
class ClothObjectCfg(AssetBaseCfg):
    """Configuration paramsters for the cloth object."""

    mesh_subfix: str = "mesh"
    """The subfix of the mesh prim path."""
    particle_system_subfix: str = "ParticleSystem"
    """The subfix of the particle system prim path."""
    class_type: type = SingleClothObject
    """The class type of the cloth object."""
