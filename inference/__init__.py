"""Closed-loop inference architecture.

while not done:
    observation = env.get_observation()           # -> canonical Observation
    action      = policy.predict_action(           # -> canonical RobotAction
                      observation, instruction)
    env.step(action)                               # apply chunk, advance sim

The inference runner is independent of training code and of any specific policy:
it only depends on the generic ``Policy`` interface and the environment's
``get_observation`` / ``step`` contract.

Two closed-loop execution patterns (documented, choose per task):
  (1) Predict N actions, execute all N, replan.
  (2) Predict N actions, execute K < N, replan.
The action chunk is consumed incrementally; K is the execution horizon.

Existing reusable LeIsaac infrastructure (KEEP):
  - scripts/evaluation/policy_inference.py        (drives a policy against an env)
  - leisaac.policy.service_policy_clients.*       (transport clients; Pi0.5 via
                                                  OpenPIServicePolicyClient)
  - leisaac.datagen.state_machine.{generate,replay}.py  (expert replay baseline)

Runtime status:
  - Inference runner seam  : IMPLEMENTED (pure Python)
  - Policy adapters        : IMPLEMENTED (Pi0.5 obs/action adapters)
  - LeIsaac inference script: VERIFIED in source (needs Isaac Sim to run)
  - Live closed-loop run   : BLOCKED by Isaac Sim runtime
"""

from __future__ import annotations

from environments.action import RobotAction
from environments.observation import Observation
from policies.base import Policy


def run_closed_loop(
    policy: Policy,
    get_observation,
    step,
    instruction: str,
    max_episode_steps: int = 300,
    execution_horizon: int | None = None,
) -> dict:
    """Generic closed-loop rollout.

    Args:
        policy: any ``Policy`` (Pi05Policy, scripted, ...).
        get_observation: callable -> Observation.
        step: callable(RobotAction) -> None (advances sim by the chunk).
        instruction: language task instruction.
        max_episode_steps: safety cap on outer loop iterations.
        execution_horizon: K in "execute K<N then replan". None = execute whole
            predicted chunk.

    Returns:
        dict with steps taken, termination reason.
    """
    policy.reset()
    steps = 0
    while steps < max_episode_steps:
        obs = get_observation()
        obs.validate()
        action: RobotAction = policy.predict_action(obs, instruction)
        action.validate()
        horizon = execution_horizon or action.action_horizon
        # A real env consumes the chunk; we expose the horizon decision here.
        step(action)
        steps += horizon
    return {"steps": steps, "reason": "max_steps", "instruction": instruction}
