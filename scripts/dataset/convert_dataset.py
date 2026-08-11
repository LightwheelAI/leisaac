#!/usr/bin/env python
"""Convert raw SO-101 episodes to a LeRobot dataset (entry-point skeleton).

Reuses LeIsaac's verified HDF5 -> LeRobot v3 converter:
  scripts/convert/isaaclab2lerobotv3.py  (lerobot==0.4.2)

Architecture: Raw Episodes (HDF5) -> Converter -> LeRobot Dataset -> Validator.

Runtime status:
  🟡 Converter VERIFIED in source.
  🔴 Live conversion BLOCKED by Isaac Sim runtime (the converter boots AppLauncher).
"""
from __future__ import annotations

import argparse
import sys

# LeIsaac converter that performs the real work (kept as the source of truth).
_LEISAAC_CONVERTER = "scripts/convert/isaaclab2lerobotv3.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert raw episodes to LeRobot")
    parser.add_argument("--hdf5_root", default="./datasets/raw")
    parser.add_argument("--repo_id", default="your_hf_username/so101_pi05_demos")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--validate", action="store_true", help="Run dataset validator after conversion")
    args = parser.parse_args()

    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[convert] BLOCKED: {exc}\nIsaac Sim runtime required by the "
              f"LeIsaac converter. See README > Known Blockers.", file=sys.stderr)
        return 2

    # TODO(dataset): delegate to the LeIsaac converter; do not reimplement.
    #   python {_LEISAAC_CONVERTER} --hdf5_root {args.hdf5_root} \
    #       --repo_id {args.repo_id} --task_name {args.task_name} --fps {args.fps}
    print(f"[convert] -> repo_id={args.repo_id} hdf5_root={args.hdf5_root}")
    print(f"[convert] Delegate to {_LEISAAC_CONVERTER} (not yet auto-invoked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
