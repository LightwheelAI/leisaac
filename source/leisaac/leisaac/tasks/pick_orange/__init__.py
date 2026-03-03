import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-PickOrange-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_env_cfg:PickOrangeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_mimic_env_cfg:PickOrangeMimicEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-Direct-v0",
    entry_point=f"{__name__}.direct.pick_orange_env:PickOrangeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.pick_orange_env:PickOrangeEnvCfg",
    },
)
