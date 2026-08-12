"""Evaluation engine.

Evaluation Runner
      ↓
Environment Reset
      ↓
Task Instruction
      ↓
Policy Inference     (generic Policy interface)
      ↓
SO-101 Execution
      ↓
Success / Failure Check   (Task contract)
      ↓
Metrics

The evaluator is model-independent: it drives any ``Policy`` over any registered
``Task``. It does not know about Pi0.5, GR00T, or training internals.

Metrics recorded per episode:
  episode_id, task, checkpoint, seed,
  success (bool), failure_reason, termination_reason,
  episode_length, completion_time, inference_latency (mean/last).

Failure reasons are structured where detectable:
  timeout | object_dropped | out_of_workspace | collision |
  joint_limit | incorrect_target | policy_divergence | sim_instability | unknown

Runtime status:
  - Evaluator + metrics       : IMPLEMENTED (pure Python, unit-tested)
  - Live multi-episode eval   : BLOCKED by Isaac Sim runtime
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from environments.action import RobotAction
from environments.observation import Observation
from policies.base import Policy
from tasks import Task


@dataclass
class EpisodeMetric:
    episode_id: int
    task: str
    checkpoint: str
    seed: int
    success: bool
    failure_reason: str = "none"
    termination_reason: str = "unknown"
    episode_length: int = 0
    completion_time_s: float = 0.0
    mean_inference_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


FAILURE_REASONS = (
    "timeout",
    "object_dropped",
    "out_of_workspace",
    "collision",
    "joint_limit",
    "incorrect_target",
    "policy_divergence",
    "sim_instability",
    "unknown",
)


@dataclass
class EvaluationReport:
    task: str
    checkpoint: str
    episodes: list[EpisodeMetric] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def to_json(self) -> str:
        return json.dumps(
            {
                "task": self.task,
                "checkpoint": self.checkpoint,
                "success_rate": self.success_rate,
                "n_episodes": len(self.episodes),
                "episodes": [e.as_dict() for e in self.episodes],
            },
            indent=2,
        )

    def to_csv(self) -> str:
        header = "episode_id,task,checkpoint,seed,success,failure_reason,termination_reason,episode_length,completion_time_s,mean_inference_latency_ms"
        rows = [header]
        for e in self.episodes:
            d = e.as_dict()
            rows.append(
                ",".join(
                    str(d[k])
                    for k in (
                        "episode_id",
                        "task",
                        "checkpoint",
                        "seed",
                        "success",
                        "failure_reason",
                        "termination_reason",
                        "episode_length",
                        "completion_time_s",
                        "mean_inference_latency_ms",
                    )
                )
            )
        return "\n".join(rows)


def evaluate_episode(
    policy: Policy,
    task: Task,
    reset_env: Callable[[], None],
    get_observation: Callable[[], Observation],
    step: Callable[[RobotAction], None],
    episode_id: int,
    checkpoint: str,
    seed: int,
    max_steps: int = 300,
) -> EpisodeMetric:
    """Run one evaluation episode and return its metric."""
    reset_env()
    task.reset(None)  # task may reconfigure object/target poses
    instruction = task.get_instruction()
    policy.reset()

    start = time.time()
    latencies: list[float] = []
    done = False
    step_i = 0
    termination = "max_steps"
    while step_i < max_steps and not done:
        t0 = time.time()
        obs = get_observation()
        obs.validate()
        action = policy.predict_action(obs, instruction)
        action.validate()
        latencies.append((time.time() - t0) * 1000.0)
        step(action)
        step_i += action.action_horizon
        if task.is_success(None):
            done = True
            termination = "success"
        elif task.is_failure(None):
            done = True
            termination = "failure"
    elapsed = time.time() - start

    success = termination == "success"
    return EpisodeMetric(
        episode_id=episode_id,
        task=task.__class__.__name__,
        checkpoint=checkpoint,
        seed=seed,
        success=success,
        failure_reason="none" if success else "unknown",
        termination_reason=termination,
        episode_length=step_i,
        completion_time_s=elapsed,
        mean_inference_latency_ms=(sum(latencies) / len(latencies) if latencies else 0.0),
    )
