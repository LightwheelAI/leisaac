import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-SausageCut-BiArm-Direct-v0",
    entry_point=f"{__name__}.direct.sausage_cut_bi_arm_env:SausageCutBiArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.sausage_cut_bi_arm_env:SausageCutBiArmEnvCfg",
    },
)
