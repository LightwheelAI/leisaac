"""Pi0.5 policy integration boundary.

Pi05Policy adapts the generic :class:`policies.base.Policy` contract to a
Physical-Intelligence OpenPI pi0.5 backend.

Two deployment shapes are supported by the OpenPI reference design (and by
LeIsaac's existing ``OpenPIServicePolicyClient``):

  A) Remote policy server (OpenPI serves pi0.5 over websocket). Pi05Policy sends
     a canonical Observation (via the observation adapter) and receives an action
     chunk. This is the shape LeIsaac already implements and what we reuse.

  B) In-process fine-tuning/inference (JAX/PyTorch). Used for training and for
     headless eval without a separate server. Wired in training/ and inference/.

This module deliberately does NOT contain imaginary model calls. Where the
exact OpenPI server message contract has not been re-verified against the
deployed server, the call is guarded and documented.

Integration status:
  - Observation adapter  : IMPLEMENTED (verified against LeIsaac joint math)
  - Action adapter       : IMPLEMENTED (verified against LeIsaac joint math)
  - Training config      : IMPLEMENTED (pi05_so101, action_dim=6)
  - Policy wrapper class : IMPLEMENTED (structure + server-client seam)
  - Live server call     : BLOCKED -- requires running OpenPI pi0.5 server
  - In-process load      : BLOCKED -- requires OpenPI installed in runtime
"""

from __future__ import annotations

from typing import Any

from environments.action import RobotAction
from environments.observation import Observation
from policies.base import PolicyWrapper
from policies.pi05.adapters import Pi05ActionAdapter, Pi05ObservationAdapter
from policies.pi05.config import Pi05TrainConfig, default_pi05_so101_config


class Pi05Policy(PolicyWrapper):
    """Generic-policy wrapper around a pi0.5 backend (server or in-process)."""

    kind = "pi05"

    def __init__(
        self,
        backend: Any | None = None,
        camera_keys: tuple[str, ...] = ("front", "wrist"),
        config: Pi05TrainConfig | None = None,
    ):
        super().__init__(backend=backend)
        self.config = config or default_pi05_so101_config()
        self._obs_adapter = Pi05ObservationAdapter(camera_keys=camera_keys)
        self._act_adapter = Pi05ActionAdapter()

    def predict_action(self, observation: Observation, instruction: str) -> RobotAction:
        observation.validate()
        # Translate canonical Observation -> OpenPI request dict.
        pi05_obs = self._obs_adapter(
            camera_images=observation.images,
            joint_state=observation.robot_state,
            task_description=instruction,
        )
        # TODO(openpi-integration): verify the exact server request/response
        # contract against the deployed OpenPI pi0.5 server before enabling.
        # LeIsaac's OpenPIServicePolicyClient sends {"images/<cam>": uint8
        # 224x224, "state": float64[6], "prompt": str} and receives
        # {"actions": (chunk, 6)}. Reuse that mapping.
        if self._backend is None:
            raise RuntimeError(
                "Pi05Policy has no backend. Start an OpenPI pi0.5 server or "
                "load an in-process model. See docs/pipeline.md > Pi0.5 Integration."
            )
        pi05_actions = self._backend.infer(pi05_obs)["actions"]  # (chunk, 6)
        return self._act_adapter(pi05_actions)
