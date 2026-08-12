#!/usr/bin/env python
"""Evaluate a pi0.5 (or any) policy on SO-101 tasks (entry-point skeleton).

Uses the model-independent evaluator (evaluation/__init__.py::evaluate_episode /
EvaluationReport) which drives any registered Task with any generic Policy.

Runtime status:
  ✅ Evaluator + metrics implemented (pure Python, unit-tested).
  🔴 Live multi-episode eval BLOCKED by Isaac Sim runtime.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate policy on SO-101 tasks")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", choices=["pi05", "scripted"], default="pi05")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./evaluation/results/report.json")
    args = parser.parse_args()

    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[evaluate] BLOCKED: {exc}\nIsaac Sim runtime required for live eval. "
              f"See README > Known Blockers.", file=sys.stderr)
        return 2

    from evaluation import EvaluationReport
    from tasks import REGISTRY
    # TODO(evaluation): build env + policy, then loop evaluate_episode over N
    # episodes and accumulate into EvaluationReport.
    report = EvaluationReport(task=args.task, checkpoint=args.checkpoint)
    print(f"[evaluate] task={args.task} policy={args.policy} "
          f"episodes={args.episodes} seed={args.seed}")
    print(f"[evaluate] Report object ready (success_rate={report.success_rate}).")
    print("[evaluate] Live loop not yet wired (needs Isaac Sim).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
