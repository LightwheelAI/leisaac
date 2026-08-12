#!/usr/bin/env python
"""Collect SO-101 demonstration episodes (entry-point skeleton).

Control source (teleoperation / scripted expert) -> Canonical Action ->
Episode Recorder -> Raw Episode (HDF5 via LeIsaac LeRobotRecorderManager).

Supported control sources (VERIFIED in LeIsaac source):
  - keyboard      (leisaac.devices.keyboard.so101_keyboard)
  - gamepad       (leisaac.devices.gamepad.so101_gamepad)
  - SO-101 leader (leisaac.devices.lerobot.so101_leader / so101_leader_remote)
  - scripted      (leisaac.datagen.state_machine.*, e.g. pick_orange)

Runtime status:
  🟡 Recorder/device infrastructure VERIFIED in source.
  🔴 Live recording BLOCKED by Isaac Sim runtime (omni.isaac.ml_archive).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SO-101 demonstrations")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--control",
        choices=["keyboard", "gamepad", "leader", "scripted"],
        default="keyboard",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="./datasets/raw")
    args = parser.parse_args()

    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[collect] BLOCKED: {exc}\nIsaac Sim runtime required for live "
              f"recording. See README > Known Blockers.", file=sys.stderr)
        return 2

    # TODO(data-collection): instantiate the chosen ControlSource + EpisodeRecorder
    # (see data_collection/__init__.py) and drive the env. Reuse LeIsaac
    # LeRobotRecorderManager for LeRobot-frame writing.
    print(f"[collect] task={args.task} control={args.control} "
          f"episodes={args.episodes} -> {args.output}")
    print("[collect] Live loop not yet wired (requires Isaac Sim).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
