#!/usr/bin/env python
"""Fine-tune / train pi0.5 on an SO-101 LeRobot dataset (entry-point).

Translates the repo-owned Pi05TrainConfig (policies/pi05/config.py) into an
OpenPI TrainConfig and launches training.

Command (when OpenPI is installed):
    python scripts/training/train_pi05.py \
        --config pi05_so101 --exp_name lift_cube_v1 \
        --dataset_repo_id your_hf_username/so101_pi05_demos

Runtime status:
  ✅ Config + OpenPI TrainConfig mapping implemented (policies/pi05/config.py).
  🔴 Live training BLOCKED: requires OpenPI runtime + GPU + a real dataset.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune pi0.5 on SO-101 data")
    parser.add_argument("--config", default="pi05_so101")
    parser.add_argument("--exp_name", default="lift_cube_v1")
    parser.add_argument("--dataset_repo_id", default="your_hf_username/so101_pi05_demos")
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    # Build the repo-owned config (no OpenPI needed for this part).
    from policies.pi05.config import default_pi05_so101_config
    cfg = default_pi05_so101_config()
    cfg.exp_name = args.exp_name
    cfg.data.repo_id = args.dataset_repo_id
    if args.resume_from:
        cfg.resume = True
        cfg.pytorch_weight_path = args.resume_from
    print(f"[train_pi05] config={cfg.name} exp={cfg.exp_name} "
          f"dataset={cfg.data.repo_id} action_dim={cfg.model.action_dim}")

    try:
        from policies.pi05.config import to_openpi_config
        openpi_cfg = to_openpi_config(cfg)
        print(f"[train_pi05] OpenPI TrainConfig built: {openpi_cfg.name}")
    except Exception as exc:  # pragma: no cover - OpenPI not installed here
        print(f"[train_pi05] BLOCKED: cannot build OpenPI config: {exc}\n"
              f"OpenPI runtime required. See README > Training Pipeline.",
              file=sys.stderr)
        return 2

    # TODO(training): invoke openpi.training entrypoint with openpi_cfg.
    print("[train_pi05] Live training launcher not yet wired (needs OpenPI + dataset).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
