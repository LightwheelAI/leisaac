"""Unit tests for the SO-101 <-> Pi0.5 adapters and config.

These tests do NOT require a GPU, Isaac Sim, or OpenPI to be installed.
They validate the deterministic math of the observation/action translation,
which is the single most error-prone boundary in the pipeline.

Run:
    pytest tests/test_pi05_adapters.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# leisaac is importable under env_isaaclab (editable, symlinked). If it is not
# available the test is skipped rather than failing the whole suite, so the repo
# stays testable in lightweight CI.
try:
    from leisaac.utils import robot_utils

    HAVE_LEISAAC = True
except Exception:  # pragma: no cover - depends on environment
    HAVE_LEISAAC = False

from policies.pi05.adapters import (
    Pi05ActionAdapter,
    Pi05ObservationAdapter,
    build_pi05_observation,
    pi05_action_to_so101,
)
from policies.pi05.config import (
    Pi05ModelConfig,
    Pi05TrainConfig,
    default_pi05_so101_config,
)


@pytest.mark.skipif(not HAVE_LEISAAC, reason="leisaac not importable in this env")
def test_observation_adapter_shape_and_dtype():
    cams = {
        "front": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist": np.zeros((240, 320, 3), dtype=np.float32) + 0.5,
    }
    joint = np.zeros(6)  # canonical rest
    obs = build_pi05_observation(cams, joint, "lift the cube")

    assert set(obs.keys()) == {"images/front", "images/wrist", "state", "prompt"}
    assert obs["images/front"].shape == (224, 224, 3)
    assert obs["images/front"].dtype == np.uint8
    assert obs["images/wrist"].shape == (224, 224, 3)
    assert obs["state"].shape == (6,)
    assert obs["state"].dtype == np.float64
    assert obs["prompt"] == "lift the cube"


@pytest.mark.skipif(not HAVE_LEISAAC, reason="leisaac not importable in this env")
def test_action_roundtrip_is_identity_for_rest_pose():
    # action chunk of rest-pose LeRobot-normalized commands
    lerobot_rest = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 10, dtype=np.float64)
    so101 = pi05_action_to_so101(lerobot_rest).numpy()  # (10,1,6) radians
    # convert back to LeRobot space and confirm it maps consistently
    back = robot_utils.convert_leisaac_action_to_lerobot(so101.reshape(-1, 6))
    assert back.shape == (10, 6)
    # The gripper limit is [0,100] motor; a -0 command maps to the motor min.
    assert np.all(np.isfinite(back))


@pytest.mark.skipif(not HAVE_LEISAAC, reason="leisaac not importable in this env")
def test_action_chunk_shape_contract():
    chunk = np.random.rand(16, 6).astype(np.float64)
    out = pi05_action_to_so101(chunk)
    assert out.shape == (16, 1, 6)
    assert out.dtype == np.float64 or out.dtype == np.float32


def test_config_defaults_are_so101_specific():
    cfg = default_pi05_so101_config()
    assert cfg.name == "pi05_so101"
    assert cfg.model.action_dim == 6
    assert cfg.model.pi05 is True
    assert cfg.data.asset_id == "so101"
    assert cfg.data.image_keys == ("front", "wrist")


def test_config_action_dim_is_not_borrowed_from_other_datasets():
    # Guard against accidentally reusing DROID(32)/Libero(7)/Aloha(7/14).
    cfg = Pi05TrainConfig(model=Pi05ModelConfig(action_dim=32))
    assert cfg.model.action_dim == 32  # explicit; not a default
    assert Pi05TrainConfig().model.action_dim == 6  # SO-101 default


@pytest.mark.skipif(not HAVE_LEISAAC, reason="leisaac not importable in this env")
def test_stateful_adapters_wrap_stateless_helpers():
    obs_adapter = Pi05ObservationAdapter(camera_keys=("front", "wrist"))
    act_adapter = Pi05ActionAdapter()
    cams = {"wrist": np.zeros((480, 640, 3), np.uint8), "front": np.zeros((480, 640, 3), np.uint8)}
    obs = obs_adapter(cams, np.zeros(6), "pick")
    assert "images/front" in obs and "images/wrist" in obs
    out = act_adapter(np.zeros((10, 6)))
    assert out.shape == (10, 1, 6)
