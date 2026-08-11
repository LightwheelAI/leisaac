#!/usr/bin/env python
"""Launch an IsaacLab SO-101 simulation (entry-point skeleton).

This is the canonical "launch simulation" entry point. It boots the Isaac Sim
app and loads an SO-101 scene/task. The actual scene/task loading reuses
LeIsaac's env registration (tasks registered with the `LeIsaac` prefix).

Runtime status:
  🔴 BLOCKED by Isaac Sim omni.isaac.ml_archive extension defect on this host.
     When Isaac Sim is repaired, this script runs via:
        isaaclab.sh -p scripts/simulation/launch_so101.py --task LiftCube
"""
from __future__ import annotations

import argparse
import sys

BLOCKER = (
    "Isaac Sim runtime unavailable on this host: the bundled Isaac Sim 5.1.0 "
    "standalone has a corrupted 'omni.isaac.ml_archive' extension, so "
    "AppLauncher(headless=True) crashes at startup (ModuleNotFoundError: "
    "omni.isaac.ml_archive). See docs/upstream.md and README > Known Blockers. "
    "Fix the Isaac Sim install (repair/reinstall the ml_archive extension) before "
    "running live simulation. The repo architecture is complete without it."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch SO-101 IsaacLab simulation")
    parser.add_argument("--task", type=str, default="LiftCube", help="Task name")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    args = parser.parse_args()

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[launch_so101] BLOCKED: {exc}\n{BLOCKER}", file=sys.stderr)
        return 2

    print(f"[launch_so101] Booting Isaac Sim for task={args.task} ...")
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=args.enable_cameras)
    simulation_app = app_launcher.app

    # TODO(sim-integration): load registered SO-101 task env and serve a render/
    # teleoperation loop. Reuse leisaac.tasks.<task>_env_cfg and the
    # LeRobotRecorderManager for data collection.
    print("[launch_so101] Sim app booted. (Live scene loading not yet wired.)")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
