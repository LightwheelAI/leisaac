from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils import configclass

from . import recorders


@configclass
class PreStepDirectEnvActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action in direct environment recorder term."""
    class_type: type[RecorderTerm] = recorders.PreStepDirectEnvActionsRecorder


@configclass
class DirectEnvActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for recording actions and states in direct environment."""
    record_pre_step_actions = PreStepDirectEnvActionsRecorderCfg()
