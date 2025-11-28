from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.utils import configclass


@configclass
class RecorderEnhanceDirectRLEnvCfg(DirectRLEnvCfg):
    """Configuration for the RecorderEnhanceDirectRLEnv."""

    recorders: object = DefaultEmptyRecorderManagerCfg()
    """reuse the recorders from the ManagerBasedEnvCfg."""

    # termination conditions
    never_time_out: bool = False
    """Whether enable time out in this env. Set it to True when teleoperating."""
    manual_terminate: bool = False
    """Whether enable manual terminate in this env. Set it to True when teleoperating."""
    return_success_status: bool = False
    """When manual_terminate is True, _get_dones() will return this value as done"""
