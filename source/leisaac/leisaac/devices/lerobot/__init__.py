"""LeRobot SO101 Leader device for SE(3) control."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bi_so101_leader import BiSO101Leader
    from .so101_leader import SO101Leader
    from .so101_leader_remote import SO101LeaderRemote

__all__ = ["BiSO101Leader", "SO101Leader", "SO101LeaderRemote"]

_LAZY_IMPORTS = {
    "BiSO101Leader": (".bi_so101_leader", "BiSO101Leader"),
    "SO101Leader": (".so101_leader", "SO101Leader"),
    "SO101LeaderRemote": (".so101_leader_remote", "SO101LeaderRemote"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
