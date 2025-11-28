from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.direct_rl_env import DirectRLEnv
from isaaclab.managers import RecorderManager
from isaacsim.core.simulation_manager import SimulationManager

from .direct_rl_env_cfg import RecorderEnhanceDirectRLEnvCfg


class RecorderEnhanceDirectRLEnv(DirectRLEnv):
    """Direct RL Environment with recorder enhancement."""

    def __init__(self, cfg: RecorderEnhanceDirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.recorders:
            self.recorder_manager = RecorderManager(self.cfg.recorders, self)
            print("[INFO] Recorder Manager: ", self.recorder_manager)

    def close(self):
        if self.cfg.recorders:
            del self.recorder_manager
        super().close()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if self.cfg.recorders:
            self.recorder_manager.reset(env_ids)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        Modified from super().step() with additional logic to use recorder_manager.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # [enhance] trigger recorder terms for pre-step calls
        if self.cfg.recorders:
            self.recorder_manager.record_pre_step()

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # [enhance] trigger recorder terms for post-step calls
        if self.cfg.recorders:
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # [enhance] trigger recorder terms for pre-reset calls
            if self.cfg.recorders:
                self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # [enhance] trigger recorder terms for post-reset calls
            if self.cfg.recorders:
                self.recorder_manager.record_post_reset(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """
        Modified from super().reset() with additional logic to use recorder_manager.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # [enhance] trigger recorder terms for pre-reset calls
        if self.cfg.recorders:
            self.recorder_manager.record_pre_reset(indices)

        self._reset_idx(indices)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # [enhance] trigger recorder terms for post-reset calls
        if self.cfg.recorders:
            self.recorder_manager.record_post_reset(indices)

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # return observations
        self.obs_buf = self._get_observations()  # [enhance] store observations in buffer
        return self.obs_buf, self.extras

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.
        DirectRLEnv don't have reset_to function, we add here, and add recorder manager support.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # [enhance] trigger recorder terms for pre-reset calls
        if self.cfg.recorders:
            self.recorder_manager.record_pre_reset(env_ids)

        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.reset_to(state, env_ids, is_relative=is_relative)
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # [enhance] trigger recorder terms for post-reset calls
        if self.cfg.recorders:
            self.recorder_manager.record_post_reset(env_ids)

        # return observations
        self.obs_buf = self._get_observations()  # [enhance] store observations in buffer
        return self.obs_buf, self.extras
