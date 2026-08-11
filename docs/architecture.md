# System Architecture

Deep-dive companion to the root [`README.md`](README.md). Describes each layer,
its boundaries, and the data-flow contracts. All claims are grounded in the
actual source (LeIsaac `24d3bcd` and the new top-level modules).

## Data-Flow Overview

```mermaid
flowchart TD
    A[SO-101 Digital Twin] --> B[IsaacLab Simulation]
    B --> C[Task / Environment]
    C --> D[Data Collection]
    D --> E[Raw Episodes]
    E --> F[LeRobot Dataset]
    F --> G[Pi0.5 Training]
    G --> H[Pi0.5 Checkpoint]
    H --> I[Closed-loop Inference]
    I --> J[SO-101 Evaluation]
    J --> K[Metrics]

    subgraph PolicyBoundary[Policy Boundary]
      G
      H
      I
    end

    C -. canonical Observation .-> POL[Generic Policy Interface]
    POL -. Pi0.5 Adapter .-> I
```

The key rule: **the simulation produces a canonical `Observation`; policies
consume it through adapters.** IsaacLab never emits Pi0.5-specific tensors.

---

## SO-101 Digital Twin (`robots/so101/`)

Single source of truth for SO-101 robot knowledge.

- `SO101_JOINT_NAMES` = `(shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper)` — order must match
  `source/leisaac/leisaac/assets/robots/lerobot.py`.
- `SO101_USD_JOINT_LIMITS_DEG`, `SO101_MOTOR_LIMITS` — USD-degree vs
  LeRobot-normalized [0,100] limits.
- `SO101RobotSpec` — frozen dataclass; `SO101_SPEC` singleton.
- `load_isaac_articulation_cfg()` — delegates to LeIsaac's `SO101_FOLLOWER_CFG`
  (one source of truth), requires Isaac Sim to import.

**Boundary:** contains NO Pi0.5 / GR00T / LeRobot-training code.

---

## Simulation Layer (`simulation/`, `scripts/simulation/`)

Responsibilities: IsaacLab app startup, physics, scene, sensors, stepping,
rendering.

- Boots via `isaaclab.sh` (use kit python: deactivate conda).
- Scene/table/objects/cameras/lighting: LeIsaac task configs
  (`source/leisaac/leisaac/tasks/<task>/<task>_env_cfg.py`).
- **Must NOT depend on** Pi0.5, GR00T, or LeRobot training.

**Boundary:** depends only on IsaacLab + robot + environment; never imports a
policy backend directly.

---

## Environment Layer (`environments/`)

Hosts the **canonical contracts**:

- `Observation` — `{images: {cam: uint8 HWC}, robot_state: (6,), timestamp,
  metadata}` with `.validate()`.
- `RobotAction` — `{joints: (H,5), gripper: (H,1)}` with `.as_vector (H,6)` and
  `.validate()`.

These are pure-Python, unit-tested, and form the translation boundary between
the simulator and any policy.

---

## Task Layer (`tasks/`)

Framework-agnostic `Task` ABC:

```python
class Task(ABC):
    def reset(self, env): ...
    def get_instruction(self) -> str: ...
    def is_success(self, env) -> bool: ...
    def is_failure(self, env) -> bool: ...   # optional
```

`TaskRegistry` lets tasks be referenced by name in configs/scripts. Upstream
LeIsaac tasks (`lift_cube`, `pick_orange`, …) implement this shape via IsaacLab
termination/observation terms + `task_description`.

---

## Data Collection (`data_collection/`)

```
ControlSource -> Canonical Action -> EpisodeRecorder -> Raw Episode
```

- `ControlSource` (ABC): teleoperator or scripted expert; produces
  `RobotAction` from an `Observation`.
- `EpisodeRecorder` (ABC): consumes `(Observation, RobotAction)`; storage-agnostic.
- Reuses LeIsaac: `LeRobotRecorderManager`, `StreamingHDF5DatasetFileHandler`,
  `devices/*` (keyboard, gamepad, lerobot SO-101 leader, lekiwi),
  `datagen/state_machine/*`.

The recorder does not care whether the action came from keyboard, gamepad,
leader, VR, script, or a policy.

---

## Dataset Layer (`dataset/`)

```
Raw Episodes (HDF5) -> Converter -> LeRobot Dataset -> Validator
```

- `RawEpisodeSchema`, `LerobotDatasetRef` define the canonical representations.
- Converter: **LeIsaac** `scripts/convert/isaaclab2lerobotv3.py`
  (lerobot==0.4.2), NOT reimplemented.
- Validator entry point: `scripts/dataset/validate_dataset.py` (runs once a
  dataset exists; checks episode/frame counts, dims, NaN/Inf, metadata).

SO-101 mapping: `front`/`wrist` images (640×480 uint8), `state` (6,), `action`
(6,), instruction = `task_description`, per-frame timestamps, one HDF5 group per
demo.

---

## Policy Interface (`policies/base/`)

Generic `Policy` ABC — the **only** policy contract the sim/eval loops depend on:

```python
class Policy(ABC):
    def reset(self): ...
    def predict_action(self, observation: Observation, instruction: str) -> RobotAction: ...
```

`PolicyWrapper` is the base for backend-adapting policies (Pi05Policy,
Gr00tPolicy). We keep our own top-level `Policy` so the public interface does
not depend on `leisaac` internals, while mirroring common policy-server
contracts. LeIsaac's own `Policy` ABC (`leisaac.policy.base`) is retained
upstream.

---

## Pi0.5 Integration (`policies/pi05/`)

```
Canonical Observation
   -> Observation Adapter  (policies/pi05/adapters.py)
   -> Pi0.5 (OpenPI server or in-process)
   -> Action Adapter       (policies/pi05/adapters.py)
   -> Canonical RobotAction -> SO-101
```

- `config.py`: `Pi05TrainConfig` (`pi05_so101`, `action_dim=6`,
  `asset_id="so101"`) mirroring OpenPI's `TrainConfig` schema; `to_openpi_config()`
  lazily builds a real OpenPI `TrainConfig`.
- `adapters.py`: deterministic joint remapping via
  `leisaac.utils.robot_utils`; produces OpenPI obs
  `{images/<cam>: uint8 224x224, state: float64[6], prompt}`; consumes
  `(chunk,6)` actions.
- `policy.py`: `Pi05Policy(PolicyWrapper)` — structure + server seam; fails
  clearly without a backend (no imaginary model calls).

**Why a dedicated SO-101 config?** OpenPI's released `pi05_*` configs use
action dims 7/14/32 — none are SO-101's 6. Reusing another's norm stats would
silently break training.

---

## Training (`training/`, `scripts/training/`)

```
LeRobot Dataset -> Pi0.5 Dataset Adapter -> OpenPI Training -> Checkpoint
```

- `TrainingJob` + `checkpoint_dir()` convention:
  `checkpoints/pi05_so101/<exp_name>/step_XXXXX/`.
- Launcher `train_pi05.py` builds the repo config, translates to OpenPI
  `TrainConfig`, then (TODO) invokes the OpenPI trainer.

---

## Inference (`inference/`, `scripts/inference/`)

```python
while not done:
    obs = get_observation()                 # canonical Observation
    action = policy.predict_action(obs, instruction)  # canonical RobotAction
    step(action)                            # apply chunk
```

`run_closed_loop()` supports "execute all N" or "execute K<N then replan"
(`execution_horizon`). Independent of training code.

---

## Evaluation (`evaluation/`, `scripts/evaluation/`)

Model-independent. `evaluate_episode` + `EvaluationReport` (JSON + CSV) drive
any `Policy` over any `Task`. Metrics: success rate, episode length, completion
time, mean inference latency, structured failure reasons
(`timeout | object_dropped | out_of_workspace | collision | joint_limit |
incorrect_target | policy_divergence | sim_instability | unknown`).

---

## Boundary Summary

| Depends on | Must NOT depend on |
|---|---|
| Robot layer | Pi0.5, GR00T, LeRobot training |
| Simulation layer | Pi0.5, GR00T, LeRobot training |
| Environment (Observation/Action) | any policy backend |
| Policy interface | IsaacLab, OpenPI internals |
| Pi0.5 adapter | the generic sim/eval loops |
| Evaluation | Pi0.5 / GR00T / training internals |
