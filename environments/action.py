"""Canonical action contract.

Policy output -> Policy Action Adapter -> Canonical SO-101 Action -> Environment
-> Robot Controller.

Keeping a robot-side action representation lets future policies (Pi0.5, GR00T,
scripted, human) be swapped without touching the controller or environment.

Runtime status: data-structure layer is pure-Python and unit-tested. Live
application to the IsaacLab articulation requires the Isaac Sim runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RobotAction:
    """Canonical SO-101 action.

    Attributes:
        joints: (5,) or (action_horizon, 5) arm joint commands in canonical
            SO-101 space (IsaacLab radians). A 2-D array represents an action
            *chunk* (multiple future steps), matching Pi0.5's chunked output.
        gripper: (1,) or (action_horizon, 1) gripper command.
        timestamp: when produced.
        metadata: free-form (policy name, chunk index, ...).
    """

    joints: np.ndarray = field(default_factory=lambda: np.zeros((1, 5), dtype=np.float32))
    gripper: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_horizon(self) -> int:
        return max(self.joints.shape[0], self.gripper.shape[0])

    @property
    def as_vector(self) -> np.ndarray:
        """Concatenate into the canonical (horizon, 6) SO-101 vector."""
        j = np.asarray(self.joints).reshape(self.action_horizon, -1)
        g = np.asarray(self.gripper).reshape(self.action_horizon, -1)
        return np.concatenate([j, g], axis=-1)

    def validate(self) -> None:
        if self.joints.shape[-1] != 5:
            raise ValueError(f"joints must be (H,5), got {self.joints.shape}")
        if self.gripper.shape[-1] != 1:
            raise ValueError(f"gripper must be (H,1), got {self.gripper.shape}")
        if self.joints.shape[0] != self.gripper.shape[0]:
            raise ValueError("joints and gripper horizons must match")
