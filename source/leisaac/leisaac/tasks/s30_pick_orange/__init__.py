import gymnasium as gym

gym.register(
    id="LeIsaac-S30-PickOrange-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s30_pick_orange_env_cfg:S30PickOrangeEnvCfg",
    },
)
