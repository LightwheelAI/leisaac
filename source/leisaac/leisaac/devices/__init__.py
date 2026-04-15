from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .device_base import DeviceBase
    from .gamepad import SO101Gamepad
    from .keyboard import S30Keyboard, SO101Keyboard
    from .lekiwi import LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader
    from .lerobot import BiSO101Leader, SO101Leader, SO101LeaderRemote

__all__ = [
    "DeviceBase",
    "S30Keyboard",
    "SO101Gamepad",
    "SO101Keyboard",
    "LeKiwiGamepad",
    "LeKiwiKeyboard",
    "LeKiwiLeader",
    "BiSO101Leader",
    "SO101Leader",
    "SO101LeaderRemote",
]

_LAZY_IMPORTS = {
    "DeviceBase": (".device_base", "DeviceBase"),
    "S30Keyboard": (".keyboard", "S30Keyboard"),
    "SO101Gamepad": (".gamepad", "SO101Gamepad"),
    "SO101Keyboard": (".keyboard", "SO101Keyboard"),
    "LeKiwiGamepad": (".lekiwi", "LeKiwiGamepad"),
    "LeKiwiKeyboard": (".lekiwi", "LeKiwiKeyboard"),
    "LeKiwiLeader": (".lekiwi", "LeKiwiLeader"),
    "BiSO101Leader": (".lerobot", "BiSO101Leader"),
    "SO101Leader": (".lerobot", "SO101Leader"),
    "SO101LeaderRemote": (".lerobot", "SO101LeaderRemote"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
