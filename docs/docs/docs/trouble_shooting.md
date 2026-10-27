# Trouble Shooting

## Vscode debugger does not work

> related issue: [IsaacLab/issues/3305](https://github.com/isaac-sim/IsaacLab/issues/3305)

When you launch the program using VSCode Python Debugger, you may encounter the following error:

```shell
OSError: libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

Please try installing the corresponding dependencies in your conda environment:
```shell
conda install -c conda-forge gcc=12 -y
```

## Isaac Sim 5.1 instability on NVIDIA drivers newer than 580

> related issue: [isaac-sim/IsaacSim#912](https://github.com/isaac-sim/IsaacSim/issues/912)

Isaac Sim 5.1 (the version LeIsaac pins to) is validated against the NVIDIA **580** driver branch.
On newer branches two distinct failure modes were observed; they should not be confused with each
other, because the workaround for one does not cover the other:

1. **Startup crash** — the app dies during RTX initialisation (access violation in `rtx.scenedb`)
   before any stage is loaded. `import isaacsim` still succeeds. Seen on the Vulkan backend on both
   Windows and Linux; on Windows forcing the D3D12 backend (`--/app/vulkan=false`) lets the app start.
2. **Camera-render crash (Windows, D3D12, windowed mode)** — the app starts fine, but the very first
   camera observation raises `CUDA error: an illegal memory access`, and torch/PhysX-fabric fail with
   the same error afterwards. This is a driver-level failure of the windowed D3D12 render path with
   `--enable_cameras`; headless (offscreen) rendering is not affected.

Startup-crash matrix, all verified on real machine:

| OS | GPU | Driver | Backend | Result |
|---|---|---|---|---|
| Windows 11 25H2 | RTX 4060 Laptop 8 GB | 610.62 | Vulkan (default) | **startup crash** (`rtx.scenedb` access violation) |
| Windows 11 25H2 | RTX 4060 Laptop 8 GB | 610.62 | D3D12 (`--/app/vulkan=false`) | starts (`Simulation App Startup Complete`) |
| Ubuntu 22.04 | (see upstream issue) | 615.71.09 | Vulkan (default) | **startup crash** (same signature) |
| Ubuntu 22.04 | (see upstream issue) | 580.126.09 | Vulkan (default) | works |

Camera-render crash matrix — Windows 11 + RTX 4060 Laptop, driver 610.62, after a successful D3D12
startup:

| Run mode | Result |
|---|---|
| Headless + `--enable_cameras` (offscreen) | works (23 min smoke, 13643 frames; end-to-end teleop: 120 env.step, 0 CUDA errors) |
| Headless without cameras | works |
| Windowed (app default viewport) + `--enable_cameras` | **crash** (CUDA illegal memory access at first camera read) |
| Windowed without `--enable_cameras` | not supported by LeIsaac camera tasks (raises "A camera was spawned without the --enable_cameras flag") |

Enabling the `performance` rendering preset (which disables all OptiX/AI denoisers), forcing serial
omni-graph execution, hiding the UI, or disabling the present thread did **not** fix the windowed
crash — the fault is inside the driver's D3D12/CUDA interop for the windowed render path.

### Working configuration without a driver downgrade: headless teleoperation

Camera data, HDF5 recording, and keyboard teleoperation all work headless. The only loss is the
on-screen simulation viewport; cameras keep rendering offscreen and are captured in the dataset.

```powershell
python scripts/environments/teleoperation/teleop_se3_agent.py `
  --task=LeIsaac-SO101-PickOrange-v0 `
  --teleop_device=keyboard `
  --num_envs=1 `
  --device=cuda `
  --enable_cameras `
  --headless `
  --kit_args="--/app/vulkan=false"
```

> note: on a truly headless run the app window that normally receives teleop keyboard events does
> not exist, so the B/W/A/S/... hotkeys do not reach the device. Pipe your teleop actions through a
> device that does not depend on the app window (e.g. a console key listener or the SO101 leader
> arm), or verify with `--record` runs only.

A quick smoke test that DOES reproduce both failure modes (unlike a bare `SimulationApp` startup,
which only covers crash mode 1):

```python
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})  # offscreen; on driver 610.62 this must print ✓
print("startup OK; closing...")
app.close()
# startup OK covers driver gate + RTX init (failure mode 1).
# Then run any LeIsaac camera task with --headless --enable_cameras to cover failure mode 2.
```

### Alternative / definitive fix: stay on the 580 driver branch

Isaac Sim 5.1 documents `580.65.06` on Linux and `580.88` on Windows. NVIDIA's validation list names
those same versions. Staying on the 580 branch avoids both failure modes entirely, at the cost of not
running the newest driver. This is the only verified way to restore **windowed + camera** operation.