import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-CleanToyTable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clean_toy_table_env_cfg:CleanToyTableEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-CleanToyTable-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clean_toy_table_bi_arm_env_cfg:CleanToyTableBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-CleanToyTable-BiArm-Direct-v0",
    entry_point=f"{__name__}.direct.clean_toy_table_bi_arm_env:CleanToyTableBiArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.clean_toy_table_bi_arm_env:CleanToyTableBiArmEnvCfg",
    },
)
