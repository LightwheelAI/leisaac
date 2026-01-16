from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders


@configclass
class PreStepDirectEnvActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action in direct environment recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepDirectEnvActionsRecorder


@configclass
class PostStepDirectEnvProcessedActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term in direct environment."""

    class_type: type[RecorderTerm] = recorders.PostStepDirectEnvProcessedActionsRecorder


@configclass
class InitialStateWithParticleObjectsRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state with particle objects recorder term."""

    class_type: type[RecorderTerm] = recorders.InitialStateWithParticleObjectsRecorder


@configclass
class InitialStateWithCuttableObjectsRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state with cuttable objects recorder term."""

    class_type: type[RecorderTerm] = recorders.InitialStateWithCuttableObjectsRecorder


@configclass
class DirectEnvActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for recording actions and states in direct environment."""

    record_pre_step_actions = PreStepDirectEnvActionsRecorderCfg()
    record_post_step_processed_actions = PostStepDirectEnvProcessedActionsRecorderCfg()


@configclass
class DirectEnvActionStateWithParticleObjectsRecorderManagerCfg(DirectEnvActionStateRecorderManagerCfg):
    """Recorder configuration for recording actions and states with particle objects in direct environment."""

    record_initial_state = InitialStateWithParticleObjectsRecorderCfg()


@configclass
class DirectEnvActionStateWithCuttableObjectsRecorderManagerCfg(DirectEnvActionStateRecorderManagerCfg):
    """Recorder configuration for recording actions and states with cuttable objects in direct environment."""

    record_initial_state = InitialStateWithCuttableObjectsRecorderCfg()
