# RL Training

The RL training module enables training manipulation policies with reinforcement learning using [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (PPO). It runs fully in simulation with parallel environments and no human teleoperation required.

## Training

```shell
python scripts/datagen/rl/train.py \
    --task LeIsaac-SO101-LiftCube-RL-v0 \
    --num_envs 512 \
    --max_iterations 1500 \
    --headless
```

<details>
<summary><strong>Parameter descriptions for train.py</strong></summary>

- `--task`: Gym task ID to train. Required.

- `--num_envs`: Number of parallel simulation environments. More environments = faster data collection. Default: `512`.

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
python scripts/datagen/rl/play.py \
    --task LeIsaac-SO101-LiftCube-RL-v0 \
    --checkpoint logs/rl/<run>/model_<iter>.pt \
    --num_envs 4 \
    --num_episodes 20
```

## Reward Design

The LiftCube RL task uses three reward terms:

| Term | Weight | Description |
|------|--------|-------------|
| `cube_success` | 100.0 | One-time bonus when cube height ≥ 20 cm above robot base. Episode ends immediately after (early termination). |
| `ee_to_cube` | 1.5 | `1 - tanh(5 × dist(TCP, cube))` — guides TCP to cube center. Range [0, 1]. |
| `cube_height` | 10.0 | `tanh(5 × max(h - 4.6 cm, 0))` — zero below 4.6 cm, monotonically increasing above. Range [0, 1]. |

**TCP (Tool Center Point)** is computed as the midpoint between the two fingertip contact surfaces, derived from `body_pos_w` and `body_quat_w` with calibrated local offsets from the USD collision mesh:

- Jaw tip offset (jaw body local frame): `(0.0, -0.05, 0.02)`
- Gripper tip offset (gripper body local frame): `(-0.012, 0.0, -0.08)`

**Termination**: episode ends on timeout (15 s) or when cube height ≥ 20 cm (success).

## PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `gamma` | 0.95 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_param` | 0.1 | PPO clip range |
| `entropy_coef` | 0.005 | Entropy regularization to encourage exploration |
| `learning_rate` | 1e-4 | Adam learning rate |
| `schedule` | adaptive | Adjusts learning rate based on KL divergence |
| `desired_kl` | 0.01 | Target KL divergence for adaptive schedule |
| `num_learning_epochs` | 5 | PPO update epochs per rollout |
| `num_mini_batches` | 4 | Mini-batches per epoch |
| `num_steps_per_env` | 100 | Rollout steps per environment per update |
| `num_envs` (recommended) | 512 | Parallel environments — more = faster sparse reward discovery |

## Action Space

RL training uses the `rl_so101leader` device mode — delta end-effector control with a binary gripper:

| Component | Dims | Description |
|-----------|------|-------------|
| `arm_action` | 6 | Delta EE pose (dx, dy, dz, droll, dpitch, dyaw), scale=(0.02, 0.02, 0.02, 0.5, 0.5, 0.5) → ±2 cm / ±0.5 rad per step |
| `gripper_action` | 1 | Binary: action > 0 → open (1.0 rad), action < 0 → close (0.2 rad) |
| **Total** | **7** | |

## Observation Space

26D flat vector (concatenated):

| Term | Dims |
|------|------|
| `joint_pos` | 6 |
| `joint_vel` | 6 |
| `ee_frame_state` (pos + quat, robot frame) | 7 |
| `cube_pos_relative_to_ee` | 3 |
| `cube_quat` (orientation in world frame) | 4 |
| **Total** | **26** |

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

4. Train with the generic script: `python scripts/datagen/rl/train.py --task LeIsaac-SO101-MyTask-RL-v0`.
