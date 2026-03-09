# RL Training

The RL training module enables training manipulation policies with reinforcement learning using [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (PPO). It runs fully in simulation with parallel environments and no human teleoperation required.

## Training

```shell
python scripts/rl/train_lift_cube.py \
    --num_envs 64 \
    --max_iterations 1500 \
    --log_dir logs/rl/lift_cube \
    --headless
```

<details>
<summary><strong>Parameter descriptions for train_lift_cube.py</strong></summary>

- `--num_envs`: Number of parallel simulation environments. More environments = faster data collection. Default: `64`.

- `--max_iterations`: Number of PPO update iterations. Default: `1500`.

- `--log_dir`: Directory to save model checkpoints and tensorboard logs. Default: `logs/rl/lift_cube`.

- `--seed`: Random seed for reproducibility. Default: `42`.

- `--headless`: Run without rendering window for faster training.

- `--device`: Computation device, such as `cpu` or `cuda`.

</details>

::::tip
Training logs (tensorboard) are written to `<log_dir>/<timestamp>/`. Monitor progress with:

```shell
tensorboard --logdir logs/rl/lift_cube
```

Key metrics to watch: `Train/mean_reward` (total episode reward) and `Episode/rew_cube_stable_hold` (stable grasp signal).
::::

## Reward Design

The LiftCube RL task uses four reward terms:

| Term | Weight | Description |
|------|--------|-------------|
| `cube_success` | 100.0 | One-time bonus when cube reaches target height (≥ 20 cm). Episode ends immediately after. |
| `ee_to_cube` | 0.5 | Gaussian reward (σ=10 cm) guiding EE toward cube. |
| `cube_stable_hold` | 5.0 | Flat reward when EE has been within 8 cm of cube for 10+ consecutive steps **and** cube is off the ground. |
| `cube_height` | 5.0 | Gaussian reward (peak at 20 cm) × hold gate. Encourages lifting the cube once it is stably grasped. |

The **hold gate** is the key mechanism: `cube_stable_hold` and `cube_height` both require the EE to be continuously near the cube for `min_hold_steps` steps, preventing rewards from knocked or bounced cubes.

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
2. Create `<task>/<task>_rl_env_cfg.py` inheriting from the base env config:

```python
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

3. Register the gym environment in `<task>/__init__.py`.
4. Write a training script following `scripts/rl/train_lift_cube.py`.
