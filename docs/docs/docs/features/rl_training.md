# RL Training

The RL training module enables training manipulation policies with reinforcement learning using [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (PPO). It runs fully in simulation with parallel environments and no human teleoperation required.

## Training

```shell
python scripts/rl/train.py \
    --task LeIsaac-SO101-LiftCube-RL-v0 \
    --num_envs 64 \
    --max_iterations 1500 \
    --headless
```

<details>
<summary><strong>Parameter descriptions for train.py</strong></summary>

- `--task`: Gym task ID to train. Required.

- `--num_envs`: Number of parallel simulation environments. More environments = faster data collection. Default: `64`.

- `--max_iterations`: Number of PPO update iterations. Default: `1500`.

- `--log_dir`: Base directory for logs. Runs are saved to `<log_dir>/<task_slug>/<timestamp>/`. Default: `logs/rl`.

- `--seed`: Random seed for reproducibility. Default: `42`.

- `--headless`: Run without rendering window for faster training.

- `--device`: Computation device, such as `cpu` or `cuda`.

</details>

::::tip
Training logs (tensorboard) are written to `logs/rl/<task_slug>/<timestamp>/`. Monitor progress with:

```shell
tensorboard --logdir logs/rl
```

Key metrics to watch: `Train/mean_reward` (total episode reward) and individual reward terms such as `Episode/rew_cube_height`.
::::

## Evaluation

```shell
python scripts/rl/play.py \
    --task LeIsaac-SO101-LiftCube-RL-v0 \
    --checkpoint logs/rl/<run>/model_<iter>.pt \
    --num_envs 4 \
    --num_episodes 20
```

## Reward Design

The LiftCube RL task uses four reward terms:

| Term | Weight | Description |
|------|--------|-------------|
| `cube_success` | 500.0 | Per-step bonus when cube height ≥ 20 cm above robot base. Episode ends immediately after. |
| `ee_to_cube` | 0.5 | `1 - tanh(5 × dist)` — guides EE toward cube. Always active. |
| `cube_grasped` | 1.0 | Both jaw and gripper sides physically contact the cube AND gripper is closed. Detected via ContactSensor. |
| `cube_height` | 2.0 | `exp(-2 × |h - 0.20|)` peaking at 20 cm. Only active when `cube_grasped > 0`. |

Grasp detection uses two `ContactSensor` instances (`jaw_contact` and `gripper_contact`) to confirm bilateral physical contact — both sides must touch the cube simultaneously. This prevents the degenerate solution of "EE near cube + gripper closed" without actually lifting.

**Termination**: episode ends on timeout (15 s) or when cube height ≥ 20 cm (success).

## Action Space

RL training uses the `rl_so101leader` device mode — delta end-effector control with a binary gripper:

| Component | Dims | Description |
|-----------|------|-------------|
| `arm_action` | 6 | Delta EE pose (dx, dy, dz, droll, dpitch, dyaw), scale=0.05 → ±5 cm/step |
| `gripper_action` | 1 | Binary: 0 = open (1.0 rad), 1 = close (0.4 rad) |
| **Total** | **7** | |

## Observation Space

22D flat vector (concatenated):

| Term | Dims |
|------|------|
| `joint_pos` | 6 |
| `joint_vel` | 6 |
| `ee_frame_state` (pos + quat, robot frame) | 7 |
| `cube_pos_relative_to_ee` | 3 |
| **Total** | **22** |

## Adding a New RL Task

1. Create `<task>/mdp/rewards.py` with reward functions.
2. Create `<task>/<task>_rl_env_cfg.py` with `TRAIN_CFG` dict and env config class:

```python
TRAIN_CFG = { ... }  # PPO hyperparameters

@configclass
class MyTaskRLEnvCfg(MyTaskEnvCfg):
    observations: MyTaskRLObsCfg = MyTaskRLObsCfg()
    rewards: MyTaskRLRewardsCfg = MyTaskRLRewardsCfg()
    terminations: MyTaskRLTerminationsCfg = MyTaskRLTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.use_teleop_device("rl_so101leader")  # or "bi_rl_so101leader" for bi-arm
        self.scene.front = None  # disable camera for faster training
        self.episode_length_s = 15.0
```

3. Register the gym environment in `<task>/__init__.py` with both `env_cfg_entry_point` and `rsl_rl_cfg_entry_point`:

```python
gym.register(
    id="LeIsaac-SO101-MyTask-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<task>_rl_env_cfg:MyTaskRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.<task>_rl_env_cfg:TRAIN_CFG",
    },
)
```

4. Train with the generic script: `python scripts/rl/train.py --task LeIsaac-SO101-MyTask-RL-v0`.
