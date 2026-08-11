"""Dataset pipeline architecture.

Raw Simulation Episodes (HDF5 / IsaacLab EpisodeData)
            ↓
Raw Dataset Schema
            ↓
Converter  (IsaacLab HDF5 -> LeRobot Dataset v3)
            ↓
LeRobot Dataset
            ↓
Dataset Validator

Existing reusable LeIsaac infrastructure (KEEP, do not rewrite):
  - scripts/convert/isaaclab2lerobotv3.py  : HDF5 -> LeRobot v3 (lerobot==0.4.2)
  - scripts/convert/isaaclab2lerobot.py    : earlier HDF5 -> LeRobot converter
  - scripts/convert/lerobot2isaaclab.py    : reverse direction
  - leisaac.enhance.datasets.lerobot_dataset_handler.LeRobotDatasetHandler
        (builds the LeRobot feature schema from an IsaacLab env)
  - leisaac.enhance.datasets.hdf5_dataset_file_handler
        (streaming HDF5 raw-episode writer)

This module defines the schema/validation seam so the canonical representation
(raw episode) is distinct from the LeRobot representation.

Mapping documented for SO-101 (verified against LeIsaac source):
  camera  "front"  -> observation.images.front  (uint8 HWC, 640x480)
  camera  "wrist"  -> observation.images.wrist  (uint8 HWC, 640x480)
  robot state      -> observation.state          (6,) radians
  action           -> action                     (6,) radians
  instruction      -> task_description string
  timestamps       -> per-frame sim time
  episodes         -> HDF5 group per demo; LeRobot episode index

Runtime status:
  - Schema/validation seam     : IMPLEMENTED (pure Python)
  - LeIsaac HDF5 converters    : VERIFIED in source (need Isaac Sim + lerobot==0.4.2)
  - Live conversion            : BLOCKED by Isaac Sim runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawEpisodeSchema:
    """Canonical raw-episode description (before LeRobot conversion)."""

    camera_keys: tuple[str, ...] = ("front", "wrist")
    image_shape: tuple[int, int, int] = (480, 640, 3)  # (h, w, c)
    state_dim: int = 6
    action_dim: int = 6
    fps: int = 30
    has_instruction: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LerobotDatasetRef:
    """Pointer to a produced LeRobot dataset."""

    repo_id: str
    root: str
    fps: int = 30
    episode_count: int = 0
    features: dict[str, Any] = field(default_factory=dict)
