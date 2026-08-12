"""SO-101 Pi0.5 training configuration.

This module is the repository-owned training config for fine-tuning
Physical Intelligence's pi0.5 on SO-101 LeRobot datasets.

Grounded in (verified against):
  - Local OpenPI checkout ~/Desktop/pi0/openpi, commit 15a9616
  - openpi.training.config.TrainConfig schema (lines 465-557 of
    src/openpi/training/config.py)
  - openpi.training.config.pi0_config.Pi0Config(pi05=True, action_horizon=...)

Key finding that drives this file
--------------------------------
The upstream OpenPI configs that exist are:
    pi05_aloha, pi05_aloha_pen_uncap, pi05_droid, pi05_droid_finetune,
    pi05_full_droid_finetune, pi05_libero
NONE of them are SO-101. They use action spaces of 7/14 (Aloha), 32 (DROID),
or 7 (Libero). SO-101 has a 6-D action space (5 arm joints + 1 gripper) and
two cameras ("front", "wrist"). Therefore a dedicated config is REQUIRED and
must NOT reuse another dataset's norm stats or action_dim.

The LeRobot <-> simulator joint remapping (radians/degrees + USD-limit to
motor-limit) is handled by leisaac.utils.robot_utils, which these adapters
wrap rather than replace (see ../adapters.py).

This module is intentionally dependency-light: it defines plain dataclasses
that mirror OpenPI's schema so the repository is inspectable and the config
can be validated (pytest) WITHOUT installing openpi. At training time
(`training/train_pi05.py`) this config is translated into a real OpenPI
`TrainConfig` via `to_openpi_config()`.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class Pi05ModelConfig:
    """Mirror of openpi.training.config.pi0_config.Pi0Config, pi05 variant.

    Only the fields we actually set for SO-101 are modeled here. The
    `to_openpi_config` helper maps these onto the real Pi0Config at runtime.
    """

    pi05: bool = True
    # SO-101 action space: 5 arm joints + 1 gripper.
    action_dim: int = 6
    # pi05 default action chunk length for single-arm; matches pi05_libero (10).
    action_horizon: int = 16
    # SO-101 is a single-arm robot; no bi-arm.
    discrete_state_input: bool = False


@dataclasses.dataclass
class Pi05DataConfig:
    """SO-101 LeRobot dataset configuration for pi0.5 fine-tuning.

    Mirrors openpi.training.config.LeRobotDROIDDataConfig semantics but
    targets an SO-101 repo_id with SO-101-specific repacking.
    """

    repo_id: str = "your_hf_username/so101_pi05_demos"
    # Norm-stats asset id. Reuses pi05_base assets dir; the asset_id selects the
    # normalization statistics bundle to load (must match the action_dim=6
    # SO-101 dataset, NOT a DROID/Libero dataset).
    assets_dir: str = "gs://openpi-assets/checkpoints/pi05_base/assets"
    asset_id: str = "so101"
    prompt_from_task: bool = True
    # Camera keys present in the SO-101 dataset. Order matters: OpenPI expects a
    # stable, ordered image dict.
    image_keys: tuple[str, ...] = ("front", "wrist")
    # SO-101 state vector = joint_pos (5) concatenated with gripper (1).
    state_dim: int = 6
    fps: int = 30


@dataclasses.dataclass
class Pi05TrainConfig:
    """Repository-owned SO-101 pi0.5 fine-tuning config.

    Field-for-field mirror of the relevant subset of
    openpi.training.config.TrainConfig so it can be validated standalone.
    """

    name: str = "pi05_so101"
    project_name: str = "so101-pi05"
    exp_name: str = "lift_cube_v1"

    model: Pi05ModelConfig = dataclasses.field(default_factory=Pi05ModelConfig)
    data: Pi05DataConfig = dataclasses.field(default_factory=Pi05DataConfig)

    # Start from the released pi05_base weights.
    weight_loader: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    pytorch_weight_path: str | None = None

    pytorch_training_precision: str = "bfloat16"
    lr_schedule_warmup_steps: int = 10_000
    lr_schedule_peak_lr: float = 5e-5
    lr_schedule_decay_steps: int = 1_000_000
    lr_schedule_decay_lr: float = 5e-5
    optimizer_clip_gradient_norm: float = 1.0
    ema_decay: float | None = 0.999

    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    num_train_steps: int = 20_000

    log_interval: int = 100
    save_interval: int = 1_000
    keep_period: int | None = 5_000

    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True

    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    policy_metadata: dict[str, Any] | None = dataclasses.field(
        default_factory=lambda: {"reset_pose": [0.0, -1.0, 0.9, 0.5, 0.0, -0.1]}
    )

    @property
    def assets_dirs(self) -> str:
        return f"{self.assets_base_dir}/{self.name}"

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.checkpoint_base_dir}/{self.name}/{self.exp_name}"


def default_pi05_so101_config() -> Pi05TrainConfig:
    """Return a fresh, default SO-101 pi0.5 training config."""
    return Pi05TrainConfig()


def to_openpi_config(cfg: Pi05TrainConfig):
    """Translate this repository config into a real OpenPI TrainConfig.

    Imports openpi lazily so this module stays importable for validation
    tests even when openpi is not installed in the active environment.

    Args:
        cfg: repository-owned SO-101 pi0.5 config.

    Returns:
        An instance of openpi.training.config.TrainConfig.
    """
    import pathlib

    from openpi.training import config as openpi_config
    from openpi.training.config import (
        AdamW,
        AssetsConfig,
        CosineDecaySchedule,
        LeRobotDROIDDataConfig,
        Pi0Config,
    )
    from openpi.training.weight_loaders import CheckpointWeightLoader

    model = Pi0Config(
        pi05=True,
        action_dim=cfg.model.action_dim,
        action_horizon=cfg.model.action_horizon,
    )
    if cfg.model.discrete_state_input:
        model = dataclasses.replace(model, discrete_state_input=True)

    # LeRobotDROIDDataConfig is OpenPI's generic custom-LeRobot-dataset factory
    # (the docstring explicitly says use it for your own converted dataset).
    # It carries repo_id + norm-stats AssetsConfig; the SO-101-specific
    # observation/action repacking transform is attached by the dataset layer.
    data = LeRobotDROIDDataConfig(
        repo_id=cfg.data.repo_id,
        assets=AssetsConfig(
            assets_dir=cfg.data.assets_dir,
            asset_id=cfg.data.asset_id,
        ),
        base_config=openpi_config.DataConfig(prompt_from_task=cfg.data.prompt_from_task),
    )

    return openpi_config.TrainConfig(
        name=cfg.name,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        model=model,
        data=data,
        weight_loader=CheckpointWeightLoader(cfg.weight_loader),
        pytorch_weight_path=cfg.pytorch_weight_path,
        pytorch_training_precision=cfg.pytorch_training_precision,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=cfg.lr_schedule_warmup_steps,
            peak_lr=cfg.lr_schedule_peak_lr,
            decay_steps=cfg.lr_schedule_decay_steps,
            decay_lr=cfg.lr_schedule_decay_lr,
        ),
        optimizer=AdamW(clip_gradient_norm=cfg.optimizer_clip_gradient_norm),
        ema_decay=cfg.ema_decay,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_train_steps=cfg.num_train_steps,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        keep_period=cfg.keep_period,
        overwrite=cfg.overwrite,
        resume=cfg.resume,
        wandb_enabled=cfg.wandb_enabled,
        policy_metadata=cfg.policy_metadata,
    )
