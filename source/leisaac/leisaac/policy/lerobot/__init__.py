import sys
import types

from . import helpers
from .helpers import *


def create_module_hierarchy(path: str):
    """
    create a module hierarchy in sys.modules
    """
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        sub_path = ".".join(parts[:i])
        if sub_path not in sys.modules:
            mod = types.ModuleType(sub_path)
            sys.modules[sub_path] = mod
            if i > 1:
                parent_path = ".".join(parts[: i - 1])
                setattr(sys.modules[parent_path], parts[i - 1], mod)


helpers_path = "lerobot.scripts.server.helpers"
create_module_hierarchy(helpers_path)

fake_lerobot_module = sys.modules[helpers_path]
fake_lerobot_module.__dict__.update(helpers.__dict__)

RemotePolicyConfig.__module__ = helpers_path
TimedObservation.__module__ = helpers_path
TimedAction.__module__ = helpers_path
