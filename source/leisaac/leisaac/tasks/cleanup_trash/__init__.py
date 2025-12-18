import gymnasium as gym

gym.register(
    id="LeIsaac-LeKiwi-CleanupTrash-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cleanup_trash_env_cfg:CleanupTrashEnvCfg",
    },
)
