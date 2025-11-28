import os

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg


def export_env(
    env_name: str,
    device: str = "cuda:0",
    num_envs: int = 1,
    teleop_device: str = "so101leader",
):
    assert (
        os.environ.get("LEISAAC_ASSETS_ROOT") is not None
    ), "should set LEISAAC_ASSETS_ROOT in os.environ when using export_env."

    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
    env_cfg.use_teleop_device(teleop_device)
    # disable recorders when export env
    env_cfg.recorders = None

    env = gym.make(env_name, cfg=env_cfg)

    return env
