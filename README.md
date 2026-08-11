# SO-101 Pi0.5 Research Pipeline

> A modular research platform for **SO-101** + **IsaacLab** simulation,
> demonstration collection, **LeRobot** dataset generation, **Pi0.5**
> (Physical Intelligence / OpenPI) fine-tuning, and closed-loop simulation
> evaluation.

This repository is an **independent derivative of
[LeIsaac](https://github.com/LightwheelAI/leisaac)** (Apache-2.0). It retains
LeIsaac's IsaacLab/SO-101/teleoperation/dataset infrastructure and re-targets
the primary learned policy to **Pi0.5**. Attribution and licensing are preserved
(see [`NOTICE`](NOTICE), [`THIRD_PARTY.md`](THIRD_PARTY.md),
[`docs/upstream.md`](docs/upstream.md)).

---

## 1. Project Objective

Build a maintainable research platform:

```
SO-101 Digital Twin
        ↓
IsaacLab Simulation
        ↓
Manipulation Task / Environment
        ↓
Demonstration Collection
        ↓
Raw Episodes
        ↓
LeRobot Dataset
        ↓
Pi0.5 Fine-tuning
        ↓
Pi0.5 Checkpoint
        ↓
Closed-loop Inference
        ↓
SO-101 Evaluation
        ↓
Success Rate / Metrics
```

The architecture deliberately decouples the **simulation** from any specific
policy: the simulator talks to a generic `Policy` interface, and Pi0.5 (or
GR00T, or a scripted expert) plugs in behind a model-specific adapter.

---

## 2. End-to-End Pipeline

```
SO-101 Digital Twin
        ↓  robots/so101
IsaacLab Simulation
        ↓  simulation/
Manipulation Task
        ↓  tasks/ + environments/
Data Collection (teleop / scripted)
        ↓  data_collection/
Raw Episodes (HDF5)
        ↓  dataset/raw
LeRobot Converter
        ↓  scripts/dataset/convert_dataset.py (LeIsaac isaaclab2lerobotv3.py)
LeRobot Dataset
        ↓  dataset/lerobot
Pi0.5 Preprocessing + Fine-tuning
        ↓  policies/pi05 + training/
Pi0.5 Checkpoint
        ↓  checkpoints/pi05_so101/<exp>/
Closed-loop Inference
        ↓  inference/ + scripts/inference/
SO-101 Evaluation
        ↓  evaluation/
Metrics (JSON / CSV)
        ↓  evaluation/results/
```

---

## 3. System Architecture

| Layer | Responsibility | Location |
|---|---|---|
| **robots** | SO-101 articulation, joints, limits, actuators, gripper, cameras | `robots/so101/` |
| **simulation** | IsaacLab startup, physics, scene, sensors, stepping | `simulation/`, `scripts/simulation/` |
| **environments** | canonical Observation / Action contracts, env scaffolding | `environments/` |
| **tasks** | task init, instruction, success/failure | `tasks/` |
| **data_collection** | control source → canonical action → recorder | `data_collection/` |
| **dataset** | raw episodes → LeRobot → validation | `dataset/`, `scripts/dataset/` |
| **policies** | generic `Policy` interface + Pi0.5 impl | `policies/base/`, `policies/pi05/` |
| **training** | Pi0.5 fine-tuning config + launcher | `training/`, `scripts/training/` |
| **inference** | closed-loop runner | `inference/`, `scripts/inference/` |
| **evaluation** | model-independent evaluator + metrics | `evaluation/`, `scripts/evaluation/` |
| **configs** | robot / sim / task / data / policy / training / eval configs | `configs/` |
| **scripts** | one entry point per pipeline stage | `scripts/<stage>/` |
| **docs** | architecture, pipeline, roadmap, upstream | `docs/` |

The **upstream LeIsaac** implementation lives under `source/leisaac/` and is
reused where appropriate (robot asset, tasks, recorders, converters, policy
clients). The new top-level modules above are the project's own clean boundary
layer; they *reference* LeIsaac rather than duplicate it.

---

## 4. Repository Structure (actual)

```
so101_pi05_research/
├── robots/so101/            # SO-101 spec (joints, limits, rest pose, cameras)
├── simulation/              # sim-layer boundary docs/skeleton
├── environments/           # canonical Observation + RobotAction contracts
├── tasks/                  # Task interface + TaskRegistry
├── data_collection/        # ControlSource + EpisodeRecorder seams
├── dataset/                # RawEpisodeSchema + LerobotDatasetRef
├── policies/
│   ├── base/               # generic Policy ABC + PolicyWrapper
│   └── pi05/               # config.py, adapters.py, policy.py
├── training/               # TrainingJob + checkpoint convention
├── inference/              # run_closed_loop()
├── evaluation/             # evaluate_episode + EvaluationReport + metrics
├── configs/                # config home (robot/sim/task/data/policy/train/eval)
├── scripts/
│   ├── simulation/launch_so101.py
│   ├── data_collection/collect_demos.py
│   ├── dataset/{convert_dataset,validate_dataset}.py
│   ├── training/train_pi05.py
│   ├── inference/run_pi05_inference.py
│   └── evaluation/evaluate_policy.py
├── tests/                  # architecture + adapter unit tests
├── docs/                   # architecture.md, pipeline.md, roadmap.md, upstream.md
├── source/leisaac/         # upstream LeIsaac (reused, not rewritten)
├── README.md
├── NOTICE
├── THIRD_PARTY.md
├── LICENSE
└── .gitignore
```

> This tree reflects what exists. Empty conceptual dirs (e.g. `simulation/`,
> `configs/<sub>`) are created as documented boundaries; their bodies are filled
> as runtime components are wired.

---

## 5. SO-101 Digital Twin

Defined in [`robots/so101/__init__.py`](robots/so101/__init__.py):

- **Asset:** `so101_follower.usd` (authoritative `ArticulationCfg` in
  `source/leisaac/leisaac/assets/robots/lerobot.py::SO101_FOLLOWER_CFG`).
- **Joints (6-DoF, canonical order):**
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- **Actuators:** STS3215 implicit actuators (arm + gripper), stiffness 17.8,
  damping 0.60, effort/velocity limit 10.
- **Joint limits (deg, USD):** see `SO101_USD_JOINT_LIMITS_DEG`.
- **Motor limits (LeRobot-normalized):** see `SO101_MOTOR_LIMITS` (gripper 0–100).
- **Cameras:** `front` (base), `wrist` (gripper), both 640×480 RGB @ 30 FPS.
- **State / action dim:** 6.

**Runtime status:** ✅ spec verified in source; 🔴 live spawn/actuation blocked
by the Isaac Sim `omni.isaac.ml_archive` defect (see §Known Blockers).

---

## 6. Simulation Environment

The simulation layer boots Isaac Sim via `isaaclab.sh` (when conda is
**deactivated**, so it uses the kit python with cu130) and loads an SO-101
scene + task. Scene/table/objects/cameras/lighting come from LeIsaac task
configs (e.g. `source/leisaac/leisaac/tasks/lift_cube/`). Reset and
domain-randomization are provided by IsaacLab managers + LeIsaac
`domain_randomization` utilities.

**Runtime status:** 🟡 VERIFIED in source; 🔴 live launch blocked by Isaac Sim.

---

## 7. Task Architecture

Tasks implement the framework-agnostic contract in
[`tasks/__init__.py`](tasks/__init__.py):

```python
class Task(ABC):
    def reset(self, env): ...
    def get_instruction(self) -> str: ...
    def is_success(self, env) -> bool: ...
    def is_failure(self, env) -> bool: ...   # optional
```

Upstream LeIsaac already provides concrete tasks under
`source/leisaac/leisaac/tasks/` (`lift_cube`, `pick_orange`, `fold_cloth`, …),
each a `ManagerBasedEnv` config with `task_description`, observation/termination
terms. **To add a task:** create a new `tasks/<name>/<name>_env_cfg.py` modeled
on `lift_cube`, register it (LeIsaac `gym` registry), and optionally register a
`Task` subclass in `tasks.REGISTRY` for the evaluator.

---

## 8. Observation Flow

```
IsaacLab sensors (front/wrist RGB, joint states)
        ↓
Canonical Observation            environments/observation.py
  { images: {front, wrist}, robot_state: (6,), timestamp, metadata }
        ↓  Pi0.5 Observation Adapter   policies/pi05/adapters.py
Pi0.5 request
  { "images/front": uint8 224x224, "images/wrist": uint8 224x224,
    "state": float64[6], "prompt": str }
```

The simulation only ever produces the canonical `Observation`. Policy adapters
translate it — the simulator contains **no Pi0.5-specific tensors**.

---

## 9. Action Flow

```
Policy output (action chunk)
        ↓  Pi0.5 Action Adapter   policies/pi05/adapters.py
Canonical RobotAction          environments/action.py
  { joints: (H,5), gripper: (H,1) }  -> as_vector (H,6)
        ↓
SO-101 controller (IsaacLab action manager)
```

Joint remapping (radians↔degrees, USD-limit↔motor-limit) is performed by
`leisaac.utils.robot_utils` (reused, not reimplemented).

---

## 10. Data Collection

```
Task → Environment → ControlSource → Canonical Action → Recorder → Raw Episode
```

**Control sources (verified in LeIsaac source):**
`keyboard`, `gamepad`, `SO-101 leader` (lerobot), `lekiwi`, and **scripted**
state-machine experts (`source/leisaac/leisaac/datagen/state_machine/`).

**Recorder:** `leisaac.enhance.managers.lerobot_recorder_manager.LeRobotRecorderManager`
writes LeRobot frames per sim step; raw HDF5 streaming via
`StreamingHDF5DatasetFileHandler`.

**Runtime status:** 🟡 infrastructure verified; 🔴 live recording blocked by
Isaac Sim.

---

## 11. Dataset Pipeline

```
Raw Episodes (HDF5 / IsaacLab EpisodeData)
        ↓  scripts/convert/isaaclab2lerobotv3.py  (LeIsaac, lerobot==0.4.2)
LeRobot Dataset (v3)
        ↓  scripts/dataset/validate_dataset.py
Validation (episode/frame counts, dims, NaN/Inf, metadata)
```

SO-101 key mapping (verified against LeIsaac source):
`front`/`wrist` images (uint8 640×480), `state` (6,), `action` (6,),
`task_description` instruction, per-frame timestamps, one HDF5 group per demo.

**Runtime status:** 🟡 converter verified in source; 🔴 live conversion blocked
by Isaac Sim. The validator entry point is runnable once a dataset exists.

---

## 12. Pi0.5 Integration

```
Canonical Observation
        ↓  Observation Adapter      policies/pi05/adapters.py
Pi0.5 (OpenPI server or in-process)
        ↓  Action Adapter          policies/pi05/adapters.py
Canonical RobotAction → SO-101
```

`Pi05Policy` ([`policies/pi05/policy.py`](policies/pi05/policy.py)) implements
the generic `Policy` and wraps the adapters. The backend (OpenPI server / model)
is injected; without it the policy fails with a clear message (no fake calls).

| Aspect | Status |
|---|---|
| Observation adapter | ✅ implemented (verified vs LeIsaac joint math) |
| Action adapter | ✅ implemented (verified vs LeIsaac joint math) |
| Training config `pi05_so101` | ✅ implemented (`action_dim=6`) |
| `to_openpi_config()` mapping | ✅ implemented (lazy OpenPI import) |
| Live server call | 🔴 blocked (needs OpenPI pi0.5 server) |
| In-process model load | 🔴 blocked (needs OpenPI runtime) |

---

## 13. Training Pipeline

```
LeRobot Dataset
        ↓  Pi0.5 dataset adapter (OpenPI LeRobotDROIDDataConfig)
OpenPI / Pi0.5 fine-tuning (JAX/PyTorch)
        ↓
Checkpoint  checkpoints/pi05_so101/<exp_name>/step_XXXXX/
```

Configured by `policies/pi05/config.py::Pi05TrainConfig` (name `pi05_so101`,
`action_dim=6`, `asset_id="so101"`, base weights `pi05_base`). The launcher
`scripts/training/train_pi05.py` builds this config and translates it to an
OpenPI `TrainConfig` via `to_openpi_config()`.

**Note:** OpenPI's released `pi05_*` configs are DROID(32)/Libero(7)/Aloha(7/14)
— **none are SO-101**, so a dedicated `action_dim=6` config is required and is
provided here. Do not reuse another dataset's norm stats.

**Runtime status:** ✅ config implemented + unit-tested; 🔴 live training
blocked (OpenPI runtime + GPU + real dataset).

---

## 14. Inference Pipeline

```python
while not done:
    observation = env.get_observation()          # canonical Observation
    action      = policy.predict_action(obs, instruction)   # canonical RobotAction
    env.step(action)                              # apply chunk, advance sim
```

Implemented in [`inference/__init__.py::run_closed_loop`](inference/__init__.py).
Supports both "execute all N" and "execute K<N then replan" via
`execution_horizon`. Independent of training code. Reuses LeIsaac's
`scripts/evaluation/policy_inference.py` pattern.

**Runtime status:** ✅ runner implemented (pure Python); 🔴 live run blocked by
Isaac Sim + OpenPI server.

---

## 15. Evaluation Pipeline

```
Evaluation Runner → Environment Reset → Task Instruction → Policy Inference
        → SO-101 Execution → Success/Failure Check → Metrics
```

Implemented in [`evaluation/__init__.py`](evaluation/__init__.py):
`evaluate_episode` + `EvaluationReport` (success rate, JSON + CSV). Model
independent — drives any `Policy` over any registered `Task`. Records per
episode: id, task, checkpoint, seed, success, failure reason, episode length,
completion time, mean inference latency.

**Runtime status:** ✅ evaluator + metrics implemented + unit-tested; 🔴 live
multi-episode eval blocked by Isaac Sim.

---

## 16. Configuration

| What | Where |
|---|---|
| Robot spec | `robots/so101/__init__.py` |
| Sim / scene | LeIsaac task env configs (`source/leisaac/.../tasks/*/`) |
| Task | `tasks/<name>/<name>_env_cfg.py` + `tasks.REGISTRY` |
| Dataset | `dataset/__init__.py` (schema/ref) |
| Policy / Pi0.5 | `policies/pi05/config.py` |
| Training | `policies/pi05/config.py::Pi05TrainConfig` |
| Evaluation | `evaluation/__init__.py` + `scripts/evaluation/evaluate_policy.py` |

A `configs/` directory is reserved as the single home for committed YAML/JSON
configs as stages are wired.

---

## 17. Running the Repository

Status legend: ✅ Verified · 🟡 Implemented (not runtime-verified) ·
🔴 Blocked by environment dependency · ⚪ Planned.

| Stage | Command | Status |
|---|---|---|
| Launch sim | `isaaclab.sh -p scripts/simulation/launch_so101.py --task LiftCube` | 🔴 Isaac Sim defect |
| Collect demos | `python scripts/data_collection/collect_demos.py --task LiftCube --control keyboard` | 🔴 Isaac Sim defect |
| Convert dataset | `python scripts/dataset/convert_dataset.py --repo_id ... --hdf5_root ./datasets/raw` | 🔴 Isaac Sim defect (LeIsaac converter) |
| Validate dataset | `python scripts/dataset/validate_dataset.py --repo_id ...` | 🟡 needs LeRobot 0.4.2 + dataset |
| Train Pi0.5 | `python scripts/training/train_pi05.py --config pi05_so101 --exp_name lift_v1 --dataset_repo_id ...` | 🔴 needs OpenPI + dataset |
| Run inference | `python scripts/inference/run_pi05_inference.py --task LiftCube --checkpoint ...` | 🔴 Isaac Sim + OpenPI |
| Evaluate | `python scripts/evaluation/evaluate_policy.py --task LiftCube --checkpoint ...` | 🔴 Isaac Sim |
| Run unit tests | `python -m pytest tests/ -v` | ✅ passes (architecture + adapters) |

**Tests that run today** (no GPU/Isaac Sim/OpenPI needed):
`tests/test_architecture.py` (10 tests) and `tests/test_pi05_adapters.py`
(structural; sim-dependent tests skip cleanly).

---

## 18. Development Status

| Component | Status |
|---|---|
| Repository Architecture | ✅ |
| SO-101 Asset / Spec | ✅ verified in source |
| IsaacLab Environment | 🟡 verified in source; 🔴 launch blocked |
| Task System | 🟡 verified in source (lift_cube, etc.) |
| Reset / Randomization | 🟡 verified in source |
| Cameras / Observations | 🟡 verified in source |
| Robot Actions | 🟡 verified in source |
| Demonstration Collection | 🟡 infrastructure verified; 🔴 live blocked |
| Raw Episodes | 🟡 verified in source |
| LeRobot Conversion | 🟡 converter verified; 🔴 live blocked |
| GR00T Decoupling | ✅ (generic Policy + Pi0.5 already primary path) |
| Pi0.5 Adapter (obs/act) | ✅ implemented |
| Pi0.5 Training Config | ✅ implemented + tested |
| Pi0.5 Training (live) | 🔴 blocked (OpenPI + dataset) |
| Pi0.5 Inference (live) | 🔴 blocked (Isaac Sim + OpenPI) |
| Evaluation Engine | ✅ implemented + tested |
| End-to-End Validation | 🔴 blocked by runtime |

---

## 19. Known Blockers

**Blocker 1 — Isaac Sim `omni.isaac.ml_archive` extension defect (this host).**
The bundled Isaac Sim 5.1.0 standalone
(`/home/botforgelabs2/isaac-sim-standalone-5.1.0-linux-x86_64`, symlinked as
`~/IsaacLab/_isaac_sim`) has a **corrupted `omni.isaac.ml_archive` extension**:
its `exts/omni.isaac.ml_archive/omni/` directory is empty (no `__init__.py`), so
`AppLauncher(headless=True)` crashes at startup with
`ModuleNotFoundError: No module named 'omni.isaac.ml_archive'`. This blocks all
simulation launches, data collection, dataset conversion, inference, and eval.
**Fix:** repair/reinstall the Isaac Sim standalone (restore the `ml_archive`
extension) — a multi-GB operation, not yet performed. Tracked, not a reason to
stop architecture work.

**Blocker 2 — OpenPI runtime not installed in `env_isaaclab`.** Pi0.5 training
and in-process inference require OpenPI (JAX/PyTorch). The repo config +
adapters are implemented and lazily import OpenPI; live training/inference
remains blocked until OpenPI is installed and a real SO-101 LeRobot dataset
exists.

**Environment notes (resolved):**
- `env_isaaclab` torch upgraded to **2.7.1+cu128** → real CUDA kernels run on
  the RTX 5060 Ti (Blackwell sm_120). numpy pinned to 1.26.4 (IsaacLab/OpenPI
  need `<2`).
- Isaac Sim's own kit python (`_isaac_sim/python.sh`) ships torch **2.12.1+cu130**
  (native Blackwell support).

---

## 20. Upstream Attribution

- **LeIsaac** (https://github.com/LightwheelAI/leisaac) — Apache-2.0. Base commit
  `24d3bcd` (v0.4.0). Robot assets, tasks, teleoperation, recorders, converters,
  policy clients.
- **IsaacLab** — BSD-3-Clause (submodule, pinned `3c6e67b`, local 2.3.0).
- **Isaac Sim** — NVIDIA Omniverse License (5.1.0-rc).
- **OpenPI** (Physical Intelligence) — Apache-2.0. `msgpack_numpy` + `image_tools`
  adapted in `source/leisaac/leisaac/policy/openpi/`.
- **LeRobot** (HuggingFace) — Apache-2.0. Dataset conversion target (v0.4.2).
- **Isaac-GR00T** (NVIDIA) — Apache-2.0. Retained as **legacy/optional** policy
  backend in `source/leisaac/leisaac/policy/gr00t/` (not the primary path).

All required license/attribution files are preserved: `LICENSE` (Apache-2.0,
unmodified), `NOTICE`, `THIRD_PARTY.md`, `docs/upstream.md`.

---

## 21. Quick Start (architecture review)

```bash
# 1. Inspect the architecture
cat README.md
cat docs/architecture.md
cat docs/pipeline.md

# 2. Run the tests that do not require a GPU / Isaac Sim / OpenPI
python -m pytest tests/ -v

# 3. Review the Pi0.5 config + adapters
cat policies/pi05/config.py
cat policies/pi05/adapters.py
```

Once Isaac Sim is repaired, the live pipeline runs through the `scripts/`
entry points above.
