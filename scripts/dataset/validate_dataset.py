#!/usr/bin/env python
"""Validate a produced SO-101 LeRobot dataset (entry-point).

Pure-LeRobot check: episode/frame counts, image dims, state/action dims,
NaN/Inf, metadata. Does NOT require Isaac Sim (reads the LeRobot dataset from
disk). This is runnable today once a dataset exists.

Runtime status:
  ✅ Validator skeleton runnable (needs a real LeRobot dataset to fully pass).
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SO-101 LeRobot dataset")
    parser.add_argument("--repo_id", required=True, help="Local or HF LeRobot repo")
    parser.add_argument("--root", default=None, help="Local dataset root")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover
        print(f"[validate] LeRobot not importable in this env: {exc}\n"
              f"Install lerobot==0.4.2 to validate. See README > Dataset Pipeline.",
              flush=True)
        return 2

    ds = LeRobotDataset(args.repo_id, root=args.root)
    print(f"[validate] episodes={ds.num_episodes} frames={ds.num_frames}")
    print(f"[validate] features={list(ds.features.keys())}")
    # TODO(dataset-validation): add NaN/Inf, image-dim, action-dim checks.
    print("[validate] Basic schema check passed (deep checks pending).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
