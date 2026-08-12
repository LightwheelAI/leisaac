#!/usr/bin/env python
"""Run closed-loop pi0.5 inference in SO-101 simulation (entry-point skeleton).

Drives an environment with a Pi05Policy via the generic inference runner
(inference/__init__.py::run_closed_loop). Reuses LeIsaac's
scripts/evaluation/policy_inference.py pattern.

Runtime status:
  ✅ Inference runner implemented (inference/__init__.py), pure Python.
  🔴 Live run BLOCKED by Isaac Sim runtime (env stepping) + OpenPI server.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pi0.5 closed-loop inference")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, help="Pi0.5 checkpoint dir")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--execution_horizon", type=int, default=None)
    args = parser.parse_args()

    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[infer] BLOCKED: {exc}\nIsaac Sim runtime required. "
              f"See README > Known Blockers.", file=sys.stderr)
        return 2

    from policies.pi05.policy import Pi05Policy
    # TODO(inference): build env, load checkpoint into Pi05Policy backend,
    # then run_closed_loop(pi05, get_observation, step, instruction).
    print(f"[infer] task={args.task} checkpoint={args.checkpoint} "
          f"episodes={args.episodes}")
    print("[infer] Live loop not yet wired (needs Isaac Sim + OpenPI server).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
