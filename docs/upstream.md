# Upstream Provenance and Divergence

This document records what this repository inherited from LeIsaac, and how it
diverges. Required by our Apache-2.0 obligation to state modifications.

## Derivation point

| Item | Value (verified) |
|---|---|
| Upstream repo | https://github.com/LightwheelAI/leisaac |
| Upstream branch | `main` |
| Base commit | `24d3bcd3f1e4585740fc79921782c41617237812` |
| Upstream package version | `0.4.0` |
| Upstream license | Apache-2.0 |
| Our development branch | `dev/pi05-so101` |
| Git remote layout | upstream LeIsaac kept as remote `upstream`; our fork will be remote `origin` |

The upstream remote is deliberately preserved (renamed to `upstream`) so that
history and attribution remain inspectable, and so upstream fixes can be pulled.

## What upstream actually provides (inspected, not assumed)

Inspection of `source/leisaac/leisaac/` found:

- `assets/robots`, `assets/scenes` — SO-101 robot and scene assets
- `tasks/` — 6 concrete tasks + a `template`:
  `assemble_hamburger`, `clean_toy_table`, `cleanup_trash`, `fold_cloth`,
  `lift_cube`, `pick_orange`
- `devices/` — `keyboard`, `gamepad`, `lerobot` (SO-101 leader arm), `lekiwi`
- `datagen/state_machine` — scripted expert data generation
- `enhance/` — `assets`, `datasets`, `envs`, `managers` extensions
- `policy/` — `base.py` (abstract `Policy`), `gr00t/`, `lerobot/`, `openpi/`,
  `service_policy_clients.py`
- `utils/` — including `constant.SINGLE_ARM_JOINT_NAMES` and
  `robot_utils.convert_leisaac_action_to_lerobot` / `convert_lerobot_action_to_leisaac`

Scripts (`scripts/`): `convert/isaaclab2lerobot.py`,
`convert/isaaclab2lerobotv3.py`, `convert/lerobot2isaaclab.py`,
`datagen/state_machine/{generate,replay}.py`, `environments/list_envs.py`,
`environments/teleoperation/{replay,teleop_se3_agent,so101_joint_state_server}.py`,
`evaluation/policy_inference.py`, `mimic/*`, `tutorials/*`.

## Critical finding that reshapes the plan

Upstream is **not** GR00T-only. It already ships:

1. An abstract `Policy` base class with a single `get_action()` method, plus
   `CheckpointPolicy`, `ZMQServicePolicy`, and `WebsocketServicePolicy`
   transports (`policy/base.py`).
2. A working **`OpenPIServicePolicyClient`** (`policy/service_policy_clients.py`)
   that already speaks the OpenPI websocket protocol, resizes images to
   `224x224` with `resize_with_pad`, and sends
   `{"images/<cam>": uint8 HWC, "state": float64[6], "prompt": str}`,
   receiving `{"actions": (chunk, 6)}`.
3. An `openpi` install extra in `pyproject.toml`.

Consequence: the requested "remove GR00T from the primary path" and "add a
generic policy interface" work is **substantially already present upstream**.
The honest remaining work is therefore *not* inventing an abstraction layer —
it is:

- making Pi0.5 (not Pi0, not GR00T) the **default** policy and config,
- verifying the OpenPI client's observation/action contract against the
  **actual** `pi05_*` configs in the OpenPI tree,
- supplying a genuine SO-101 Pi0.5 **training config**, which upstream does not
  provide (upstream only has an inference-time client),
- building the evaluation/metrics layer.

Per the working rules ("do not blindly rewrite working infrastructure",
"do not create unnecessary abstraction layers"), the existing `Policy` ABC and
OpenPI transport are retained rather than replaced.

## Pi0.5 configs available in the local OpenPI checkout

Local OpenPI checkout: `~/Desktop/pi0/openpi`, commit `15a9616`.
`grep` of `src/openpi/training/config.py` for real `pi05*` config names yields:

```
pi05_aloha
pi05_aloha_pen_uncap
pi05_droid
pi05_droid_finetune
pi05_full_droid_finetune
pi05_libero
```

There is **no** SO-101 Pi0.5 config upstream — one must be authored. Note also
that upstream's client class is named `OpenPIServicePolicyClient` and is
policy-agnostic at the transport level: Pi0 vs Pi0.5 is decided by which config
and checkpoint the **server** serves, not by the client. This is documented so
we do not conflate Pi0 and Pi0.5.

## Modifications made by this repository

Tracked incrementally in git history on `dev/pi05-so101`. Summary is maintained
here as work lands; see `git log --oneline` for the authoritative record.

- Established independent project identity, `NOTICE`, `THIRD_PARTY.md`, this file.
