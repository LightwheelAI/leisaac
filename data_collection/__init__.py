"""Data collection architecture.

Control source  ->  Canonical Action  ->  Recorder
-------------------------------------------------
Teleoperator        (keyboard / gamepad / SO-101 leader / lekiwi)
ScriptedPolicy      (state-machine expert, e.g. pick_orange)
Policy              (Pi05Policy / Gr00tPolicy for replay demos)

The recorder does NOT care where the action came from. It consumes a canonical
RobotAction each step and appends it to a raw episode.

Existing reusable LeIsaac infrastructure (KEEP):
  - leisaac.enhance.managers.lerobot_recorder_manager.LeRobotRecorderManager
        (IsaacLab RecorderManager subclass; writes LeRobot frames per step)
  - leisaac.enhance.datasets.hdf5_dataset_file_handler.StreamingHDF5DatasetFileHandler
        (raw HDF5 episode stream)
  - leisaac.devices.*  (keyboard, gamepad, lerobot SO-101 leader, lekiwi)
  - scripts/datagen/state_machine/{generate,replay}.py  (scripted expert)

This module provides the canonical interface so new control sources plug in
without touching the recorder.

Runtime status:
  - Interface + recorder seam : IMPLEMENTED (pure Python)
  - LeIsaac recorder/manager   : VERIFIED in source (needs Isaac Sim to run)
  - Teleoperation devices      : VERIFIED in source (SO-101 leader keyboard/gamepad)
  - Live demo recording        : BLOCKED by Isaac Sim runtime (omni.isaac.ml_archive)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from environments.action import RobotAction
from environments.observation import Observation


class ControlSource(ABC):
    """Produces canonical RobotActions given the current observation.

    Implemented by teleoperators and scripted/scripted-expert policies.
    """

    @abstractmethod
    def get_action(self, observation: Observation) -> RobotAction:
        ...

    def reset(self) -> None:
        return None


class EpisodeRecorder(ABC):
    """Consumes canonical actions + observations and stores a raw episode."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def record_step(self, observation: Observation, action: RobotAction) -> None:
        ...

    @abstractmethod
    def finalize(self) -> dict:
        """Return episode metadata (id, length, instruction, paths)."""

    def __iter__(self) -> Iterator[tuple[Observation, RobotAction]]:
        # Optional: default recorders may not support replay.
        raise NotImplementedError
