# Autonomous Policy Guide: State Machine & RL Training

## Overview

This document covers two approaches to running the robot without human teleoperation:

| Approach | How it works | Primary use |
|---|---|---|
| **State Machine** | Scripted IK-based policy; action = EE pose (7D) + binary gripper | Data collection for imitation learning |
| **RL Training** | Learned policy via PPO; action = raw joint angles (6D continuous) | End-to-end reinforcement learning |

```
scripts/environments/state_machine/
├── pick_orange.py      # Runner script: recording (PickOrange task)
└── replay.py           # Replay script for state-machine demonstrations

scripts/rl/
└── train_pick_orange.py    # PPO training script (PickOrange task)

source/leisaac/leisaac/state_machine/
├── base.py             # StateMachineBase abstract class
└── pick_orange.py      # PickOrangeStateMachine

source/leisaac/leisaac/rl/
├── __init__.py
└── rsl_rl_wrapper.py   # IsaacLab → rsl_rl VecEnv bridge wrapper
```

---

## State Machine

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

## RL Training

### Quick Start

```bash
# Short smoke-test (4 envs, 2 iterations)
python scripts/rl/train_pick_orange.py \
    --num_envs 4 --num_iters 2 --headless

# Full training run
python scripts/rl/train_pick_orange.py \
    --num_envs 64 --num_iters 1000 --headless \
    --log_dir ./logs/pick_orange_rl
```

### How It Works

```
train_pick_orange.py
  └── gym.make("LeIsaac-SO101-PickOrange-RL-v0", cfg=env_cfg)
       └── env_cfg.use_teleop_device("so101_joint_pos")  # direct joint pos, no IK
  └── IsaaclabRslRlVecEnvWrapper(env)     # bridge to rsl_rl VecEnv API
  └── OnPolicyRunner(wrapped_env, TRAIN_CFG, log_dir, device)
  └── runner.learn(num_learning_iterations)
```

The wrapper bridges IsaacLab's `ManagerBasedRLEnv` to the TensorDict-based `rsl_rl` VecEnv interface:
- `get_observations()` → `TensorDict({"policy": obs_tensor})`
- `step(actions)` → `(TensorDict, rew, dones, extras)` with `extras["time_outs"]`

### Action & Observation Format

**Action space — `so101_joint_pos` (6D continuous):**

| Device | Dims | Layout |
|---|---|---|
| `so101_joint_pos` | 6D | `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` — raw joint angles (rad) |

> Unlike the state machine, the RL policy outputs joint angles directly with no IK in the loop.

**Observation space — PickOrange RL (28D, flat vector):**

| Term | Dims | Description |
|---|---|---|
| `joint_pos` | 6 | Current joint positions (rad) |
| `joint_vel` | 6 | Current joint velocities (rad/s) |
| `ee_frame_state` | 7 | EE position (3) + quaternion (4) in robot base frame |
| `oranges_pos_relative_to_ee` | 9 | 3 oranges × 3D position relative to EE |
| **Total** | **28** | Concatenated flat vector (`concatenate_terms=True`) |

### Reward Design

| Term | Weight | Description |
|---|---|---|
| `ee_to_nearest_orange` | 0.5 | Shaped: `exp(-2·dist)` to nearest ungrasped orange |
| `orange_grasped` × 3 | 1.0 each | Per-orange: +1 every step while orange is held |
| `orange_placed` × 3 | 2.0 each | Per-orange: +1 every step while orange is on plate |
| `task_complete_bonus` | 10.0 | Sparse: +1 when all 3 oranges are on the plate |

### Key Differences from State Machine

| | State Machine | RL |
|---|---|---|
| Action space | 8D (EE pose + binary grip) | 6D (raw joint angles) |
| Control | IK → joint angles | Direct joint position |
| Gripper | Binary (open/close signal) | Continuous joint angle |
| Device type | `so101_state_machine` | `so101_joint_pos` |
| Cameras | Required for recording | Not used (proprioceptive only) |

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

### 2. Joint Damping (State Machine only)

The IK controller requires higher damping than the robot's default (0.6 N·m·s/rad) for stable, smooth trajectories. Damping is set to **10.0 N·m·s/rad** every step via:

```python
# In PickOrangeStateMachine.get_action():
robot = env.scene["robot"]
robot.write_joint_damping_to_sim(damping=10.0)
```

This must also be applied during **replay** to match recording conditions. The state-machine replay script (`scripts/environments/state_machine/replay.py`) calls `apply_damping(env, task_type)` before every `env.step()`.

> RL training does not require this — direct joint position control is inherently stable without extra damping.

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
| `scripts/rl/train_pick_orange.py` | PPO RL training for PickOrange |
| `source/leisaac/leisaac/state_machine/base.py` | `StateMachineBase` abstract class |
| `source/leisaac/leisaac/state_machine/pick_orange.py` | `PickOrangeStateMachine` |
| `source/leisaac/leisaac/rl/rsl_rl_wrapper.py` | IsaacLab → rsl_rl VecEnv wrapper |
| `source/leisaac/leisaac/tasks/pick_orange/pick_orange_rl_env_cfg.py` | RL env config (obs, rewards, action) |
| `replay_pick_orange.sh` | Shell wrapper for replay.py |
| `record_pick_orange.sh` | Shell wrapper for pick_orange recording |
