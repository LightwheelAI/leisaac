"""Training layer.

LeRobot Dataset
      ↓
Pi0.5 Dataset Adapter
      ↓
OpenPI / Pi0.5 Training (JAX or PyTorch)
      ↓
Checkpoint

The repository-owned training config lives in policies/pi05/config.py
(Pi05TrainConfig, pi05_so101, action_dim=6). This module defines the launcher
seam and checkpoint convention.

Existing reusable LeIsaac infrastructure: none for pi0.5 training (upstream
only ships an inference-time OpenPI client). OpenPI's own training entrypoint
(openpi/training) is the backend; we translate our config into an OpenPI
TrainConfig in policies/pi05/config.py::to_openpi_config.

Checkpoint convention:
  ./checkpoints/pi05_so101/<exp_name>/step_XXXXX/
  Training also emits assets (norm stats) under ./assets/pi05_so101/.

Runtime status:
  - Config (pi05_so101)        : IMPLEMENTED + unit-tested (config shape)
  - OpenPI TrainConfig mapping : IMPLEMENTED (to_openpi_config, lazy import)
  - Launch script seam         : IMPLEMENTED (training/train_pi05.py)
  - Live training              : BLOCKED -- requires OpenPI runtime + GPU +
                                 a real LeRobot SO-101 dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingJob:
    """Description of a single pi0.5 fine-tuning job."""

    config_name: str = "pi05_so101"
    exp_name: str = "lift_cube_v1"
    dataset_repo_id: str = "your_hf_username/so101_pi05_demos"
    # Path to a checkpoint to resume from (optional).
    resume_from: str | None = None
    extra: dict[str, Any] = dict()


# Canonical checkpoint path builder (does not create files).
def checkpoint_dir(config_name: str, exp_name: str, base: str = "./checkpoints") -> str:
    return f"{base}/{config_name}/{exp_name}"
