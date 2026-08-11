"""Generic policy interface.

The simulation / evaluation loop interacts ONLY with this interface, never with
OpenPI or GR00T directly. Pi0.5 and (legacy) GR00T implement it; a scripted
expert or human teleoperator can also implement it for data collection.

Design note: LeIsaac already ships a ``Policy`` ABC in
``leisaac.policy.base`` whose single method is ``get_action``. This repository
deliberately provides its own top-level ``Policy`` so the project's public
interface does not depend on the ``leisaac`` package internals, and so the
contract can use our canonical Observation/RobotAction types. We intentionally
do NOT invent a second incompatible abstraction; the method names
(``reset`` / ``predict_action``) mirror common policy-server contracts.

Runtime status: interface is pure-Python and unit-tested. Concrete model
loading / inference requires the respective runtime (OpenPI / Isaac Sim).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from environments.observation import Observation
from environments.action import RobotAction


class Policy(ABC):
    """Model-independent policy contract used by sim + evaluation loops."""

    #: Human-readable policy kind, e.g. "pi05", "gr00t", "scripted".
    kind: str = "base"

    @abstractmethod
    def reset(self) -> None:
        """Clear any hidden state / action-chunk history."""

    @abstractmethod
    def predict_action(self, observation: Observation, instruction: str) -> RobotAction:
        """Predict the next SO-101 action chunk for ``observation``.

        Args:
            observation: canonical Observation from the environment.
            instruction: language task instruction for this episode.

        Returns:
            A RobotAction (may be an action chunk of length >= 1).
        """

    def step(self, observation: Observation, instruction: str) -> RobotAction:
        """Convenience: alias for ``predict_action`` (kept for readability)."""
        return self.predict_action(observation, instruction)


class PolicyWrapper(Policy):
    """Base class for adapters that wrap a backend policy client.

    Concrete subclasses (Pi05Policy, Gr00tPolicy) translate the canonical
    Observation -> backend-specific request and the backend response ->
    canonical RobotAction.
    """

    kind = "wrapper"

    def __init__(self, backend: Any | None = None):
        self._backend = backend

    @property
    def backend(self) -> Any:
        if self._backend is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no backend configured. "
                "See docs/pipeline.md for setup."
            )
        return self._backend

    def reset(self) -> None:
        # Default: stateless policies need no reset. Subclasses may override.
        return None
