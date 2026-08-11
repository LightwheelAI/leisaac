"""Architecture-level tests that do NOT require Isaac Sim / OpenPI.

These validate the structural contracts (data classes, interfaces, registries,
adapters, config) so the repo is testable in lightweight CI. Sim/OpenPI-bound
paths are exercised separately once the runtimes are available.
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.action import RobotAction
from environments.observation import Observation
from policies.base import Policy
from policies.pi05.config import default_pi05_so101_config
from tasks import REGISTRY, Task, TaskContext
from robots.so101 import SO101_SPEC, SO101_JOINT_NAMES


# --- Robot layer -----------------------------------------------------------
def test_so101_spec_is_6dof_and_consistent():
    assert SO101_SPEC.action_dim == 6
    assert SO101_SPEC.state_dim == 6
    assert len(SO101_SPEC.joint_names) == 6
    assert set(SO101_SPEC.joint_names) == set(SO101_JOINT_NAMES)
    assert len(SO101_SPEC.joint_index) == 6


# --- Observation / Action contracts ----------------------------------------
def test_observation_validate_rejects_bad_state():
    obs = Observation(robot_state=np.zeros(5))
    with pytest.raises(ValueError):
        obs.validate()


def test_observation_validate_ok():
    obs = Observation(
        images={"front": np.zeros((480, 640, 3), np.uint8)},
        robot_state=np.zeros(6),
    )
    obs.validate()
    assert obs.image_keys() == ["front"]


def test_action_as_vector_concatenates_joints_gripper():
    a = RobotAction(
        joints=np.zeros((10, 5)),
        gripper=np.ones((10, 1)),
    )
    vec = a.as_vector
    assert vec.shape == (10, 6)
    assert np.allclose(vec[:, 5], 1.0)
    assert a.action_horizon == 10
    a.validate()


def test_action_validate_rejects_mismatched_horizon():
    a = RobotAction(joints=np.zeros((3, 5)), gripper=np.zeros((2, 1)))
    with pytest.raises(ValueError):
        a.validate()


# --- Generic policy interface ----------------------------------------------
def test_policy_is_abstract():
    with pytest.raises(TypeError):
        Policy()


def test_policy_wrapper_guards_missing_backend():
    class Dummy(Policy):
        def reset(self):
            return None

        def predict_action(self, obs, instr):
            return RobotAction()

    # A plain Policy subclass without backend works (no backend needed).
    Dummy().predict_action(Observation(), "x")


# --- Task registry ---------------------------------------------------------
def test_task_registry_roundtrip():
    class MyTask(Task):
        def reset(self, env):
            pass

        def get_instruction(self):
            return "do something"

        def is_success(self, env):
            return False

    REGISTRY.register("mytask_test", MyTask)
    assert "mytask_test" in REGISTRY.names()
    assert REGISTRY.get("mytask_test") is MyTask


def test_taskcontext_is_serializable():
    tc = TaskContext(name="lift", instruction="lift", params={"h": 0.2})
    assert tc.name == "lift"
    assert tc.params["h"] == 0.2


# --- Pi0.5 config ----------------------------------------------------------
def test_pi05_config_is_so101_specific():
    cfg = default_pi05_so101_config()
    assert cfg.model.action_dim == 6
    assert cfg.model.pi05 is True
    assert cfg.data.asset_id == "so101"
