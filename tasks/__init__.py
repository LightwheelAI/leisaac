"""Task interface.

A Task owns task-specific behaviour: initial scene/object configuration,
the language instruction, and success/failure detection. It does NOT own the
robot, the scene rig, or the physics -- those come from the environment layer.
This keeps tasks pluggable without rewriting the robot or environment.

Conceptually (adapted to the IsaacLab ManagerBasedEnv the repo already uses):

    class Task:
        def reset(self, env): ...
        def get_instruction(self) -> str: ...
        def is_success(self, env) -> bool: ...
        def is_failure(self, env) -> bool: ...

The canonical upstream tasks (lift_cube, pick_orange, ...) implement this shape
via IsaacLab termination/observation terms and a ``task_description`` field on
the env config. We provide a thin, framework-agnostic TaskProtocol so new tasks
can be authored against a stable contract and the evaluator can drive any task.

Runtime status: interface is pure-Python and unit-tested. Live success/failure
checks run inside the Isaac Sim environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskContext:
    """Lightweight, serializable description of a task instance."""

    name: str
    instruction: str
    # Optional structured config (object poses, target poses, seeds, ...).
    params: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Framework-agnostic task contract."""

    @abstractmethod
    def reset(self, env: Any) -> None:
        """Configure the scene/objects for a new episode in ``env``."""

    @abstractmethod
    def get_instruction(self) -> str:
        """Return the language instruction for this task."""

    @abstractmethod
    def is_success(self, env: Any) -> bool:
        """Return True when the task success condition is met."""

    def is_failure(self, env: Any) -> bool:
        """Return True on a terminal failure (optional; default never fails)."""
        return False


class TaskRegistry:
    """Registry so tasks can be referenced by name (string) in configs/scripts."""

    def __init__(self) -> None:
        self._tasks: dict[str, type[Task]] = {}

    def register(self, name: str, task_cls: type[Task]) -> None:
        if name in self._tasks:
            raise ValueError(f"Task {name!r} already registered")
        self._tasks[name] = task_cls

    def get(self, name: str) -> type[Task]:
        return self._tasks[name]

    def names(self) -> list[str]:
        return sorted(self._tasks)


# Global registry used across the repo.
REGISTRY = TaskRegistry()
