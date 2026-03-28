# RL Training

The RL training module enables training manipulation policies with reinforcement learning using [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (PPO). It runs fully in simulation with parallel environments and no human teleoperation required.

:::note
End-to-end RL for manipulation is challenging — reward design, exploration, and sim-to-real transfer all require significant task-specific tuning. Currently only the **LiftCube** task is supported. Support for additional tasks will be added in future updates.
:::

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

- `--num_envs`: Number of parallel simulation environments. More environments = faster data collection. Default: value from task config.

- `--max_iterations`: Number of PPO update iterations. Default: value from agent config.

- `--seed`: Random seed for reproducibility. Default: value from agent config.

- `--headless`: Run without rendering window for faster training.

- `--device`: Computation device, such as `cpu` or `cuda`.

</details>

::::tip
Training logs (tensorboard) are written to `logs/rsl_rl/<experiment_name>/<timestamp>/`. Monitor progress with:

```shell
tensorboard --logdir logs/rsl_rl
```

Key metrics to watch: `Train/mean_reward` (total episode reward) and individual reward terms such as `Episode/rew_cube_height`.
::::

## Evaluation & Recording

Evaluate a checkpoint visually (no recording):

```shell
python scripts/datagen/rl/play.py \
    --task LeIsaac-SO101-LiftCube-RL-v0 \
    --checkpoint logs/rsl_rl/lift_cube_rl/<run>/model_<iter>.pt \
    --num_envs 1
```

Save all episodes to HDF5 (both success and failure) by adding `--record`:

```shell
python scripts/datagen/rl/play.py \
    --task LeIsaac-SO101-LiftCube-RL-v0 \
    --checkpoint logs/rsl_rl/lift_cube_rl/<run>/model_<iter>.pt \
    --num_envs 1 \
    --num_episodes 100 \
    --record --dataset_file ./datasets/rl_eval.hdf5
```

<details>
<summary><strong>Parameter descriptions for play.py</strong></summary>

- `--task`: Gym task ID. Required.

- `--checkpoint`: Path to a saved model checkpoint (`.pt`). Required.

- `--num_envs`: Number of parallel environments. Default: value from task config.

- `--num_episodes`: Total episodes to run across all envs. `0` = run indefinitely. Default: `0`.

- `--seed`: Random seed. Default: value from agent config.

- `--record`: Enable HDF5 recording. Both successful and failed episodes are saved.

- `--resume_recording`: Append to an existing dataset file instead of creating a new one.

- `--dataset_file`: Output HDF5 file path. Default: `./datasets/rl_eval.hdf5`.

- `--real-time`: Slow down simulation to real-time speed.

</details>

## Reward Design

The LiftCube RL task uses three reward terms:

| Term | Weight | Description |
|------|--------|-------------|
| `cube_success` | 200.0 | One-time bonus when cube height ≥ 20 cm above robot base. Episode ends immediately after (early termination). |
| `ee_to_cube` | 1.5 | `1 - tanh(5 × dist(TCP, cube))` — guides TCP to cube center. Range [0, 1]. |
| `cube_height` | 10.0 | `tanh(3 × max(h - 4.6 cm, 0))` — zero below 4.6 cm, monotonically increasing above. Range [0, 1]. |

**TCP (Tool Center Point)** is computed as the midpoint between the two fingertip contact surfaces, derived from `body_pos_w` and `body_quat_w` with calibrated local offsets from the USD collision mesh:

- Jaw tip offset (jaw body local frame): `(0.0, -0.05, 0.02)`
- Gripper tip offset (gripper body local frame): `(-0.012, 0.0, -0.08)`

**Termination**: episode ends on timeout (15 s) or when cube height ≥ 20 cm (success).

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
2. Create `<task>/<task>_rl_env_cfg.py` with the RL env config class:

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

3. Create `<task>/rl_agents/rsl_rl_ppo_cfg.py` with the PPO runner config:

```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class MyTaskRLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 1500
    save_interval = 50
    experiment_name = "my_task_rl"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

4. Register the gym environment in `<task>/__init__.py`:

```python
from . import rl_agents

gym.register(
    id="LeIsaac-SO101-MyTask-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<task>_rl_env_cfg:MyTaskRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rl_agents.__name__}.rsl_rl_ppo_cfg:MyTaskRLPPORunnerCfg",
    },
)
```

5. Train:

```bash
python scripts/datagen/rl/train.py \
    --task LeIsaac-SO101-MyTask-RL-v0 \
    --num_envs 512 \
    --headless
```
