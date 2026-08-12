# Implementation Roadmap

Ten phases. Each item shows DONE / IN PROGRESS / BLOCKED / TODO. This is the
authoritative progress tracker (kept in sync with the README development
status table).

Legend: ✅ DONE · 🟡 IN PROGRESS / implemented-not-runtime-verified ·
🔴 BLOCKED (runtime) · ⚪ TODO.

---

## PHASE 1 — Architecture
- ✅ Modular SO-101 research repository structure
- ✅ Canonical Observation / Action contracts (`environments/`)
- ✅ Generic `Policy` interface (`policies/base/`)
- ✅ `Task` interface + `TaskRegistry` (`tasks/`)
- ✅ Pi0.5 policy boundary (`policies/pi05/`)
- ✅ Data-collection / dataset / training / inference / evaluation seams
- ✅ Entry-point skeletons (`scripts/`)
- ✅ README + docs (architecture / pipeline / roadmap / upstream)

## PHASE 2 — SO-101 Simulation
- 🟡 Articulation cfg verified in source (`SO101_FOLLOWER_CFG`)
- 🟡 Scene / cameras / lighting verified in source
- 🔴 Live spawn / actuation (Isaac Sim `ml_archive` defect)

## PHASE 3 — Environment and Tasks
- 🟡 ManagerBasedEnv + reset + randomization verified in source
- 🟡 Tasks: lift_cube, pick_orange, fold_cloth, … verified in source
- 🔴 Live env launch (Isaac Sim defect)

## PHASE 4 — Data Collection
- 🟡 Teleoperation devices (keyboard/gamepad/leader/lekiwi) verified in source
- 🟡 Scripted state-machine experts verified in source
- 🟡 Recorder (LeRobotRecorderManager) verified in source
- 🔴 Live recording (Isaac Sim defect)

## PHASE 5 — LeRobot Dataset
- 🟡 HDF5→LeRobot converter verified in source (isaaclab2lerobotv3.py)
- 🟡 Raw/LeRobot schema + validator seam implemented
- 🔴 Live conversion (Isaac Sim defect)
- ⚪ Deep dataset validation (NaN/Inf/dim checks) — TODO once dataset exists

## PHASE 6 — Pi0.5 Integration
- ✅ Observation adapter (verified vs LeIsaac joint math)
- ✅ Action adapter (verified vs LeIsaac joint math)
- ✅ Training config `pi05_so101` (action_dim=6) + `to_openpi_config`
- 🔴 Live server call (OpenPI server not running)
- 🔴 In-process model load (OpenPI not installed in env)

## PHASE 7 — Training
- ✅ Config + OpenPI TrainConfig mapping implemented + tested
- 🔴 Live fine-tuning (OpenPI runtime + GPU + real dataset)
- ⚪ Resume / distributed / LoRA options — TODO (verify in OpenPI)

## PHASE 8 — Inference
- ✅ `run_closed_loop` implemented (pure Python)
- 🔴 Live closed-loop run (Isaac Sim + OpenPI server)

## PHASE 9 — Evaluation
- ✅ `evaluate_episode` + `EvaluationReport` (JSON/CSV) implemented + tested
- 🔴 Live multi-episode eval (Isaac Sim)
- ⚪ Generalization sweeps (object/target/seed/lighting variation) — TODO

## PHASE 10 — End-to-End Validation
- 🔴 Full pipeline (all live stages) blocked by Isaac Sim + OpenPI runtime
- ⚪ Success-rate report across tasks — TODO

---

## Next Concrete Actions (unblock order)

1. **Repair Isaac Sim** `omni.isaac.ml_archive` extension → unblocks Phases 2–5,
   8, 9 (simulation, data, conversion, inference, eval).
2. **Install OpenPI** in the training env → unblocks Phase 6 (live) + 7.
3. **Collect a small SO-101 dataset** → enables real conversion + training smoke.
4. **Run the full pipeline** end-to-end and fill Phase 10.

Until (1) and (2) are done, the architecture, interfaces, adapters, config, and
unit tests are complete and inspectable.
