import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-AssembleHamburger-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assemble_hamburger_bi_arm_env_cfg:AssembleHamburgerBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-AssembleHamburger-BiArm-Direct-v0",
    entry_point=f"{__name__}.direct.assemble_hamburger_bi_arm_env:AssembleHamburgerBiArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.assemble_hamburger_bi_arm_env:AssembleHamburgerBiArmEnvCfg",
    },
)
