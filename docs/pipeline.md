# End-to-End Research Pipeline

Lifecycle of one research iteration, from simulation launch to metrics.
Each stage lists Input → Process → Output, the responsible module/config/script,
and current implementation status.

Legend: ✅ Verified · 🟡 Implemented (not runtime-verified) · 🔴 Blocked by
environment · ⚪ Planned.

---

## 1. Launch Simulation

- **Input:** task name (e.g. `LiftCube`).
- **Process:** boot Isaac Sim via `isaaclab.sh`, load SO-101 scene + task env.
- **Output:** live IsaacLab environment (SO-101 + workspace + cameras).
- **Module:** `scripts/simulation/launch_so101.py`, LeIsaac task env configs.
- **Status:** 🔴 blocked by Isaac Sim `omni.isaac.ml_archive` defect.

## 2. Create Task

- **Input:** task definition (objects, target, success condition).
- **Process:** author `tasks/<name>/<name>_env_cfg.py` on the
  `lift_cube` template; optionally register a `Task` in `tasks.REGISTRY`.
- **Output:** registered task.
- **Module:** `tasks/`, `source/leisaac/leisaac/tasks/template/`.
- **Status:** 🟡 verified in source (lift_cube, pick_orange, etc.).

## 3. Reset Environment

- **Input:** env handle.
- **Process:** IsaacLab reset + domain randomization (objects, cameras).
- **Output:** fresh episode state.
- **Module:** IsaacLab managers + LeIsaac `domain_randomization`.
- **Status:** 🟡 verified in source.

## 4. Collect Demonstration

- **Input:** control source (keyboard / gamepad / leader / scripted) + instruction.
- **Process:** `ControlSource.get_action(obs)` → `RobotAction` → env.step.
- **Output:** per-step (Observation, Action).
- **Module:** `data_collection/`, `source/leisaac/leisaac/devices/*`,
  `source/leisaac/leisaac/datagen/state_machine/*`.
- **Status:** 🟡 infrastructure verified; 🔴 live blocked by Isaac Sim.

## 5. Record Episode

- **Input:** (Observation, Action) stream.
- **Process:** `EpisodeRecorder.record_step` → HDF5 / LeRobot frame.
- **Output:** raw episode.
- **Module:** `data_collection/EpisodeRecorder`, LeIsaac
  `LeRobotRecorderManager`, `StreamingHDF5DatasetFileHandler`.
- **Status:** 🟡 verified in source; 🔴 live blocked.

## 6. Convert Dataset

- **Input:** raw HDF5 episodes.
- **Process:** `scripts/convert/isaaclab2lerobotv3.py` (LeIsaac, lerobot==0.4.2).
- **Output:** LeRobot Dataset (v3).
- **Module:** `scripts/dataset/convert_dataset.py`, `dataset/`.
- **Status:** 🟡 converter verified; 🔴 live blocked by Isaac Sim.

## 7. Validate Dataset

- **Input:** LeRobot dataset (repo_id / local root).
- **Process:** episode/frame counts, image dims, state/action dims, NaN/Inf,
  metadata checks.
- **Output:** validation report.
- **Module:** `scripts/dataset/validate_dataset.py`, `dataset/`.
- **Status:** 🟡 validator runnable once a dataset exists (needs LeRobot 0.4.2).

## 8. Configure Pi0.5

- **Input:** dataset repo_id, exp_name, base weights (`pi05_base`).
- **Process:** build `Pi05TrainConfig` (`action_dim=6`, `asset_id="so101"`).
- **Output:** training config.
- **Module:** `policies/pi05/config.py`, `scripts/training/train_pi05.py`.
- **Status:** ✅ implemented + tested.

## 9. Train / Fine-tune

- **Input:** LeRobot dataset + config.
- **Process:** `to_openpi_config()` → OpenPI `TrainConfig` → OpenPI trainer.
- **Output:** Pi0.5 checkpoint.
- **Module:** `training/`, `scripts/training/train_pi05.py`.
- **Status:** 🔴 blocked (OpenPI runtime + GPU + real dataset).

## 10. Load Checkpoint

- **Input:** checkpoint dir.
- **Process:** load into Pi05Policy backend (server or in-process).
- **Output:** ready `Policy`.
- **Module:** `policies/pi05/policy.py`.
- **Status:** 🔴 blocked (OpenPI runtime).

## 11. Run Inference

- **Input:** env + instruction + checkpoint.
- **Process:** `run_closed_loop(policy, get_observation, step, instruction)`.
- **Output:** episode rollout (robot motions).
- **Module:** `inference/`, `scripts/inference/run_pi05_inference.py`.
- **Status:** ✅ runner implemented; 🔴 live blocked (Isaac Sim + OpenPI).

## 12. Evaluate

- **Input:** task + checkpoint + policy.
- **Process:** `evaluate_episode` over N seeds/episodes.
- **Output:** `EvaluationReport` (success rate, JSON/CSV).
- **Module:** `evaluation/`, `scripts/evaluation/evaluate_policy.py`.
- **Status:** ✅ evaluator implemented + tested; 🔴 live blocked (Isaac Sim).

## 13. Generate Metrics

- **Input:** `EvaluationReport`.
- **Output:** `evaluation/results/report.json`, `report.csv`.
- **Module:** `evaluation/EvaluationReport.to_json/to_csv`.
- **Status:** ✅ implemented.

---

## Pipeline Invariants

1. Simulation never emits policy-specific tensors — only canonical
   `Observation`.
2. Actions flow Policy → Adapter → Canonical `RobotAction` → controller.
3. The evaluator is model-independent (works with Pi05Policy, scripted, future
   policies).
4. GR00T remains a legacy/optional `Policy` implementation; Pi0.5 is primary.
