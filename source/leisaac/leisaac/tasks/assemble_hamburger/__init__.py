import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-AssembleHamburger-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assemble_hamhurger_env_cfg:AssembleHamburgerEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-AssembleHamburger-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assemble_hamhurger_bi_arm_env_cfg:AssembleHamburgerBiArmEnvCfg",
    },
)
