import torch
from collections.abc import Sequence

from leisaac.enhance.envs.manager_based_rl_leisaac_mimic_env import ManagerBasedRLLeIsaacMimicEnv


class LiftCubeMimicEnv(ManagerBasedRLLeIsaacMimicEnv):
    """
    Environment for the lift cube task with mimic environment.
    """

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        pass
