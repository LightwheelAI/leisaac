from isaaclab.managers.recorder_manager import RecorderTerm


class PreStepDirectEnvActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in direct env in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.actions
