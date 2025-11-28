from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm


class PreStepDirectEnvActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in direct env in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.actions


class InitialStateWithParticleObjectsRecorder(RecorderTerm):
    """Recorder term that records the initial state with particle objects."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        state = self._env.scene.get_state(is_relative=True)
        state["particle_object"] = dict()
        for asset_name, particle_object in self._env.scene.particle_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = particle_object.root_pose_w
            asset_state["root_pose"][:, :3] -= self._env.scene.env_origins
            state["particle_object"][asset_name] = asset_state

        return "initial_state", extract_env_ids_values(state)
