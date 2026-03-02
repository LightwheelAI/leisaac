"""Wrapper that bridges an IsaacLab ManagerBasedRLEnv to the rsl_rl VecEnv interface."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv

from rsl_rl.env import VecEnv


class IsaaclabRslRlVecEnvWrapper(VecEnv):
    """Wraps a :class:`ManagerBasedRLEnv` so it satisfies rsl_rl's :class:`VecEnv` ABC.

    The RL env cfg must configure its policy observation group with
    ``concatenate_terms = True`` so that ``obs_dict["policy"]`` is already a
    ``(num_envs, obs_dim)`` tensor rather than a dict of named terms.

    Args:
        env: An already-constructed :class:`ManagerBasedRLEnv` instance.
    """

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # VecEnv required attributes
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_actions(self) -> int:
        return self.env.action_manager.total_action_dim

    @property
    def max_episode_length(self) -> int:
        return int(self.env.max_episode_length)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    @property
    def device(self) -> torch.device | str:
        return self.env.device

    @property
    def cfg(self):
        return self.env.cfg

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        """Return current observations as a TensorDict with key ``"policy"``."""
        obs_dict = self.env.observation_manager.compute()
        return self._to_tensordict(obs_dict)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment.

        Returns:
            obs: TensorDict with ``"policy"`` key.
            rewards: Shape ``(num_envs,)``.
            dones: Float tensor ``(num_envs,)``, 1 on episode end.
            extras: Dict with at minimum ``"time_outs"`` key (bool tensor).
        """
        obs_dict, rew, terminated, timeouts, extras = self.env.step(actions)
        obs = self._to_tensordict(obs_dict)
        dones = (terminated | timeouts).float()
        extras["time_outs"] = timeouts
        return obs, rew, dones, extras

    def reset(self) -> TensorDict:
        obs_dict, _ = self.env.reset()
        return self._to_tensordict(obs_dict)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _to_tensordict(self, obs_dict: dict) -> TensorDict:
        """Convert the IsaacLab obs dict to a TensorDict.

        When ``concatenate_terms=True`` (required for RL), ``obs_dict["policy"]``
        is already a ``(num_envs, obs_dim)`` tensor.  We wrap it in a TensorDict
        so rsl_rl can access it by group name.
        """
        policy_obs = obs_dict["policy"]
        if isinstance(policy_obs, dict):
            # Fallback: concatenate manually if cfg has concatenate_terms=False
            tensors = []
            for v in policy_obs.values():
                t = v.float() if v.dtype == torch.bool else v
                tensors.append(t.reshape(self.num_envs, -1))
            policy_obs = torch.cat(tensors, dim=-1)
        return TensorDict({"policy": policy_obs}, batch_size=[self.num_envs])
