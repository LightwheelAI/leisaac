"""Canonical observation contract.

The simulation produces a *model-independent* observation. Policy-specific
adapters (see policies/pi05/adapters.py) translate this into whatever the
policy (Pi0.5, GR00T, scripted) expects.

Architectural rule (from the project spec):

    IsaacLab
       ↓
    Canonical Observation   <-- defined here
       ↓
    Policy-specific Adapter  <-- policies/*/adapters.py

NOT: IsaacLab -> Pi0.5 tensors everywhere.

Runtime status: data-structure layer is pure-Python and unit-tested. Live
population from IsaacLab sensors requires the Isaac Sim runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """Canonical SO-101 observation.

    Attributes:
        images: {camera_name: uint8 HWC (h, w, 3)}. Keys match
            robots.so101.SO101_CAMERA_KEYS by convention ("front", "wrist").
        robot_state: (state_dim,) joint positions + gripper in the canonical
            SO-101 representation (IsaacLab radians).
        timestamp: seconds (float). May be simulation time.
        metadata: free-form dict (task name, episode id, env id, ...).
    """

    images: dict[str, np.ndarray] = field(default_factory=dict)
    robot_state: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def image_keys(self) -> list[str]:
        return sorted(self.images.keys())

    def validate(self) -> None:
        """Raise if the contract is violated (used by tests and recorders)."""
        if self.robot_state.shape[-1] != 6:
            raise ValueError(f"robot_state must be 6-dim, got {self.robot_state.shape}")
        for key, img in self.images.items():
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"image {key!r} must be HWC uint8, got {img.shape}")
