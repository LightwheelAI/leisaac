# Third-Party Attribution

All entries below were verified against the actual files vendored in or depended
on by this repository. Versions/commits are recorded as observed, not assumed.

## Primary upstream: LeIsaac

- **Project:** LeIsaac (LeRobot & IsaacLab Integration)
- **Source:** https://github.com/LightwheelAI/leisaac
- **License:** Apache License 2.0 (see `LICENSE`, retained unmodified)
- **Base commit:** `24d3bcd3f1e4585740fc79921782c41617237812` (package version `0.4.0`)
- **Author (upstream `pyproject.toml`):** Yinghao Shuai <yinghao.shuai@lightwheel.ai>
- **What is retained:** IsaacLab task/environment infrastructure
  (`source/leisaac/leisaac/tasks`), SO-101 robot + scene assets
  (`.../assets`), teleoperation devices (`.../devices`: keyboard, gamepad,
  lerobot/so101 leader, lekiwi), dataset conversion
  (`scripts/convert/isaaclab2lerobot*.py`), state-machine datagen, and the
  policy client layer (`.../policy`).

Apache-2.0 obligations honoured: `LICENSE` retained verbatim, `NOTICE` added
stating the derivation and upstream copyright, modifications documented in
`docs/upstream.md`. Existing per-file copyright headers are left intact.

## Vendored / interfaced third-party code

| Project | Where | License | Verified detail |
|---|---|---|---|
| IsaacLab | `dependencies/IsaacLab` (git submodule) | BSD-3-Clause | Submodule pinned to `3c6e67bb5c7ada942a6d1884ab69338f57596f77`; local checkout at `~/IsaacLab` reports `VERSION` = `2.3.0` at the same commit. |
| Isaac Sim | bundled with IsaacLab | NVIDIA Omniverse License | Local `_isaac_sim/VERSION` = `5.1.0-rc.19+release.26219.9c81211b.gl`. LeIsaac `pyproject.toml` classifier declares `Isaac Sim :: 5.1.0`. |
| OpenPI (Physical Intelligence) | interfaced via `.../policy/openpi/` | Apache-2.0 | `msgpack_numpy.py` + `image_tools.py` are adapted from OpenPI. Upstream client targets OpenPI commit `5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d`. |
| msgpack-numpy | `.../policy/openpi/msgpack_numpy.py` | BSD-3-Clause | File docstring credits https://github.com/lebedov/msgpack-numpy as the adaptation source. |
| LeRobot (HuggingFace) | `.../policy/lerobot/`, dataset conversion | Apache-2.0 | Upstream client targets LeRobot `v0.3.3`; `pyproject.toml` optional extra pins `lerobot==0.4.2`. |
| Isaac-GR00T (NVIDIA) | `.../policy/gr00t/` (demoted to optional) | Apache-2.0 | N1.5 client targets commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`; N1.6 client targets `e8e625f4f21898c506a1d8f7d20a289c97a52acf`. |

## Note on GR00T

GR00T is **retained** as an optional policy backend, not deleted. Its code is
Apache-2.0 and its attribution above remains binding for as long as the files
are present. The re-targeting work moves GR00T off the *primary* policy path;
it does not strip its licensing.
