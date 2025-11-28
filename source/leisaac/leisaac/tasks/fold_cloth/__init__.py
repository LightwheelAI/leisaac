import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-FoldCloth-BiArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fold_cloth_bi_arm_env_cfg:FoldClothBiArmEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-FoldCloth-BiArm-Direct-v0",
    entry_point=f"{__name__}.direct.fold_cloth_bi_arm_env:FoldClothBiArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.fold_cloth_bi_arm_env:FoldClothBiArmEnvCfg",
    },
)
