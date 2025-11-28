from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders


@configclass
class PreStepDirectEnvActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action in direct environment recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepDirectEnvActionsRecorder


@configclass
class InitialStateWithParticleObjectsRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state with particle objects recorder term."""

    class_type: type[RecorderTerm] = recorders.InitialStateWithParticleObjectsRecorder


@configclass
class DirectEnvActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for recording actions and states in direct environment."""

    record_pre_step_actions = PreStepDirectEnvActionsRecorderCfg()


@configclass
class DirectEnvActionStateWithParticleObjectsRecorderManagerCfg(DirectEnvActionStateRecorderManagerCfg):
    """Recorder configuration for recording actions and states with particle objects in direct environment."""

    record_initial_state = InitialStateWithParticleObjectsRecorderCfg()
