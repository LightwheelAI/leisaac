# State Machine: Recording & Replay Guide

## Overview

The state machine module provides automated data collection for manipulation tasks without human teleoperation. It runs a scripted policy, records demonstrations to HDF5 datasets, and supports replaying those demonstrations.

```
scripts/environments/state_machine/
├── pick_orange.py      # Runner script: recording (PickOrange task)
└── replay.py           # Replay script for state-machine demonstrations

source/leisaac/leisaac/state_machine/
├── base.py             # StateMachineBase abstract class
└── pick_orange.py      # PickOrangeStateMachine
```

---

## Recording

### Quick Start

```bash
# Single-arm pick-orange (3 oranges → plate)
python scripts/environments/state_machine/pick_orange.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --num_demos 1
```

> **Note — Grasp Success Rate:** In the default scene configuration the oranges are placed relatively far from the robot's end-effector, which results in a low grasp success rate. Adjusting the orange spawn positions in the task's environment config file (e.g. moving them closer to the robot base) significantly improves the success rate.

### How It Works

```
Runner script
  └── gym.make(task, cfg=env_cfg)           # create env
       └── env_cfg.use_teleop_device(       # configure IK action manager
               "so101_state_machine")
  └── sm = PickOrangeStateMachine(...)
  └── Main loop:
       actions = sm.get_action(env)         # state machine computes 8D IK action
       env.step(actions)                    # steps sim + recorder captures data
       sm.advance()                         # advance state machine
```

The runner calls `env.step(actions)` directly with the state machine's output tensor.
`preprocess_device_action()` is **not** called (that is only used in the teleoperation pipeline).

### Action Format

| Device | Dims | Layout |
|---|---|---|
| `so101_state_machine` | 8D | `[pos(3), quat(4), gripper(1)]` in robot base frame |
| `bi_so101_state_machine` | 16D | `[left_pos(3), left_quat(4), left_grip(1), right_pos(3), right_quat(4), right_grip(1)]` |

IK targets are expressed in the **robot base local frame**, not world frame:
```python
diff_w = target_pos_w - base_pos_w
target_pos_local = quat_apply(quat_inv(base_quat_w), diff_w)
```

---

## Dataset Structure

Episodes are stored in HDF5 format under the `data/` group:

```
data/
├── demo_0      # EMPTY — artifact of the initial env.reset() at startup
│   └── initial_state        # only field; num_samples=0; no actions
├── demo_1      # First real demonstration
│   ├── actions              # (T, 8) — IK pose targets passed to env.step()
│   ├── processed_actions    # (T, 6) — joint targets computed by IK solver
│   ├── initial_state        # scene state at episode start (for reset_to)
│   ├── states/              # per-step articulation/sensor states
│   └── obs/                 # per-step observations (images, joint_pos, etc.)
├── demo_2      # Second real demonstration
...
```

**Important:** `demo_0` is always empty. The **K-th recorded demonstration** is stored as `demo_K`.

When replaying, use `--select_episodes K` to load `demo_K`:
```bash
--select_episodes 1   # → demo_1, the first real episode
--select_episodes 2   # → demo_2, the second real episode
```

---

## Replay

### Quick Start

```bash
# replay_pick_orange.sh wrapper (default: demo_3 of dataset_test.hdf5)
bash replay_pick_orange.sh

# explicit call
python scripts/environments/state_machine/replay.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --task_type so101_state_machine \
    --select_episodes 1 \
    --device cuda \
    --enable_cameras \
    --replay_mode action
```

### Replay Modes

| Mode | Data replayed | Use case |
|---|---|---|
| `action` | `HDF5["actions"]` (8D IK pose targets) | IK-based devices (`so101_state_machine`) |
| `state` | `HDF5["states"]["articulation"]["robot"]["joint_position"]` | Joint-position devices (`so101leader`) |

For `so101_state_machine`, **only `action` mode is valid**. The IK action manager expects an 8D pose target; passing raw 6D joint positions (state mode) would cause a dimension mismatch.

---

## Technical Details

### 1. Gravity Disable (Two Steps Required)

IsaacLab's `disable_gravity` flag in `ArticulationCfg.spawn.rigid_props` only writes to the articulation root prim. Individual link prims each carry their own `PhysicsRigidBodyAPI` with gravity still enabled.

**Step 1** — Config level (in `use_teleop_device()`):
```python
self.scene.robot.spawn.rigid_props.disable_gravity = True
```

**Step 2** — USD stage traversal (in runner script, after `gym.make()`):
```python
_stage = omni.usd.get_context().get_stage()
for _prim in _stage.Traverse():
    if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)
```

Both steps must be present. The same pattern applies to bi-arm tasks (`"Robot"` matches both `Left_Robot` and `Right_Robot`).

### 2. Joint Damping

The IK controller requires higher damping than the robot's default (0.6 N·m·s/rad) for stable, smooth trajectories. Damping is set to **10.0 N·m·s/rad** every step via:

```python
# In PickOrangeStateMachine.get_action():
robot = env.scene["robot"]
robot.write_joint_damping_to_sim(damping=10.0)
```

This must also be applied during **replay** to match recording conditions. The state-machine replay script (`scripts/environments/state_machine/replay.py`) calls `apply_damping(env, task_type)` before every `env.step()`.

### 3. Episode Numbering

The IsaacLab recorder saves an initial-state-only episode (`num_samples=0`) on the very first `env.reset()` call (before any steps). This becomes `demo_0`.

| `--select_episodes N` | Episode loaded | Content |
|---|---|---|
| 0 | `demo_0` | Empty (no actions) — causes `TypeError` |
| 1 | `demo_1` | 1st real demonstration |
| K | `demo_K` | K-th real demonstration |

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/environments/state_machine/pick_orange.py` | Recording runner for PickOrange |
| `scripts/environments/state_machine/replay.py` | State-machine replay (with damping) |
| `source/leisaac/leisaac/state_machine/base.py` | `StateMachineBase` abstract class |
| `source/leisaac/leisaac/state_machine/pick_orange.py` | `PickOrangeStateMachine` |
| `replay_pick_orange.sh` | Shell wrapper for replay.py |
| `record_pick_orange.sh` | Shell wrapper for pick_orange recording |
