# 自主策略指南：状态机与 RL 训练

## 概述

本文档涵盖两种无需人工遥操作的机器人自主运行方式：

| 方式 | 原理 | 主要用途 |
|---|---|---|
| **状态机** | 脚本化 IK 策略；动作 = 末端位姿 (7D) + 二值夹爪 | 采集模仿学习数据 |
| **RL 训练** | 基于 PPO 的学习策略；动作 = 原始关节角 (6D 连续) | 端到端强化学习 |

```
scripts/environments/state_machine/
├── pick_orange.py      # 录制 Runner 脚本（拾橙任务）
└── replay.py           # 状态机演示回放脚本

scripts/rl/
└── train_pick_orange.py    # PPO 训练脚本（拾橙任务）

source/leisaac/leisaac/state_machine/
├── base.py             # StateMachineBase 抽象基类
└── pick_orange.py      # PickOrangeStateMachine

source/leisaac/leisaac/rl/
├── __init__.py
└── rsl_rl_wrapper.py   # IsaacLab → rsl_rl VecEnv 桥接包装器
```

---

## 状态机

### 快速开始

```bash
# 单臂拾橙（3 个橘子 → 盘子）
python scripts/environments/state_machine/pick_orange.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --num_demos 1
```

> **注意 — 夹取成功率：** 默认场景配置中橙子离机械臂末端较远，导致夹取成功率较低。在任务的环境 cfg 文件中调整橙子的生成位置（例如将其移近机械臂底座），可以显著提升夹取成功率。

### 工作原理

```
Runner 脚本
  └── gym.make(task, cfg=env_cfg)           # 创建环境
       └── env_cfg.use_teleop_device(       # 配置 IK action manager
               "so101_state_machine")
  └── sm = PickOrangeStateMachine(...)
  └── 主循环:
       actions = sm.get_action(env)         # 状态机计算 8D IK 动作
       env.step(actions)                    # 推进仿真 + 录制器采集数据
       sm.advance()                         # 推进状态机
```

Runner 脚本直接将状态机输出的动作张量传给 `env.step(actions)`，**不经过** `preprocess_device_action()`（后者仅在遥操作流程中使用）。

### 动作格式

| 设备 | 维度 | 格式 |
|---|---|---|
| `so101_state_machine` | 8D | `[pos(3), quat(4), gripper(1)]`，机械臂 base 局部坐标系 |
| `bi_so101_state_machine` | 16D | `[左_pos(3), 左_quat(4), 左_grip(1), 右_pos(3), 右_quat(4), 右_grip(1)]` |

IK 目标位置需表示在**机械臂 base 局部坐标系**下，而非世界坐标系：

```python
diff_w = target_pos_w - base_pos_w
target_pos_local = quat_apply(quat_inv(base_quat_w), diff_w)
```

---

## 数据集结构

演示数据以 HDF5 格式存储在 `data/` 组中：

```
data/
├── demo_0      # 空 episode —— 初始 env.reset() 产生的副产物
│   └── initial_state        # 仅有此字段；num_samples=0；无 actions
├── demo_1      # 第一条真实演示
│   ├── actions              # (T, 8) —— 传入 env.step() 的 IK pose 目标
│   ├── processed_actions    # (T, 6) —— IK 求解器算出的关节目标位置
│   ├── initial_state        # episode 开始时的场景状态（用于 reset_to）
│   ├── states/              # 每步的关节/传感器状态
│   └── obs/                 # 每步的观测（图像、关节位置等）
├── demo_2      # 第二条真实演示
...
```

**重要：** `demo_0` 永远是空的。**第 K 条录制的演示**存储为 `demo_K`。

回放时用 `--select_episodes K` 加载 `demo_K`：

```bash
--select_episodes 1   # → demo_1，第一条真实演示
--select_episodes 2   # → demo_2，第二条真实演示
```

---

## 回放

### 快速开始

```bash
# replay_pick_orange.sh 封装脚本（默认：dataset_test.hdf5 的 demo_3）
bash replay_pick_orange.sh

# 显式调用
python scripts/environments/state_machine/replay.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --task_type so101_state_machine \
    --select_episodes 1 \
    --device cuda \
    --enable_cameras \
    --replay_mode action
```

### 回放模式

| 模式 | 回放的数据 | 适用场景 |
|---|---|---|
| `action` | `HDF5["actions"]`（8D IK pose 目标） | IK 控制设备（`so101_state_machine`） |
| `state` | `HDF5["states"]["articulation"]["robot"]["joint_position"]` | 关节位置控制设备（`so101leader`） |

对 `so101_state_machine` 而言，**只有 `action` 模式有效**。IK action manager 期望 8D pose 输入，而 `state` 模式传入的是 6D 关节位置，会导致维度不匹配。

---

## RL 训练

### 快速开始

```bash
# 快速冒烟测试（4 个环境，2 次迭代）
python scripts/rl/train_pick_orange.py \
    --num_envs 4 --num_iters 2 --headless

# 完整训练
python scripts/rl/train_pick_orange.py \
    --num_envs 64 --num_iters 1000 --headless \
    --log_dir ./logs/pick_orange_rl
```

### 工作原理

```
train_pick_orange.py
  └── gym.make("LeIsaac-SO101-PickOrange-RL-v0", cfg=env_cfg)
       └── env_cfg.use_teleop_device("so101_joint_pos")  # 直接关节角，无 IK
  └── IsaaclabRslRlVecEnvWrapper(env)     # 桥接至 rsl_rl VecEnv 接口
  └── OnPolicyRunner(wrapped_env, TRAIN_CFG, log_dir, device)
  └── runner.learn(num_learning_iterations)
```

包装器将 IsaacLab 的 `ManagerBasedRLEnv` 桥接到基于 TensorDict 的 `rsl_rl` VecEnv 接口：
- `get_observations()` → `TensorDict({"policy": obs_tensor})`
- `step(actions)` → `(TensorDict, rew, dones, extras)`，其中 `extras["time_outs"]` 必须存在

### 动作与观测格式

**动作空间 — `so101_joint_pos`（6D 连续）：**

| 设备 | 维度 | 格式 |
|---|---|---|
| `so101_joint_pos` | 6D | `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`，原始关节角（rad） |

> 与状态机不同，RL policy 直接输出关节角，中间没有 IK 环节。

**观测空间 — PickOrange RL（28D 平坦向量）：**

| 项 | 维度 | 说明 |
|---|---|---|
| `joint_pos` | 6 | 当前关节位置（rad） |
| `joint_vel` | 6 | 当前关节速度（rad/s） |
| `ee_frame_state` | 7 | 末端位置 (3) + 四元数 (4)，机械臂 base 坐标系 |
| `oranges_pos_relative_to_ee` | 9 | 3 个橙子 × 3D 位置，相对于末端 |
| **合计** | **28** | 拼接为平坦向量（`concatenate_terms=True`） |

### 奖励设计

| 项 | 权重 | 说明 |
|---|---|---|
| `ee_to_nearest_orange` | 0.5 | 稠密：`exp(-2·dist)` 到最近未放置橙子 |
| `orange_grasped` × 3 | 各 1.0 | 逐橙子：持握期间每步 +1 |
| `orange_placed` × 3 | 各 2.0 | 逐橙子：在盘中期间每步 +1 |
| `task_complete_bonus` | 10.0 | 稀疏：全部 3 个橙子放入盘中时 +1 |

### 与状态机的关键区别

| | 状态机 | RL |
|---|---|---|
| 动作空间 | 8D（末端位姿 + 二值夹爪） | 6D（原始关节角） |
| 控制方式 | IK → 关节角 | 直接关节位置控制 |
| 夹爪 | 二值信号（开/闭） | 连续关节角 |
| 设备类型 | `so101_state_machine` | `so101_joint_pos` |
| 摄像头 | 录制时必需 | 不使用（纯本体感知） |

---

## 技术细节

### 1. 重力禁用（两步缺一不可）

IsaacLab 的 `disable_gravity` 标志只写入关节根 prim，各子 link prim 各自带有 `PhysicsRigidBodyAPI`，重力默认仍然开启。

**第一步** —— 配置层（在 `use_teleop_device()` 中）：
```python
self.scene.robot.spawn.rigid_props.disable_gravity = True
```

**第二步** —— USD Stage 遍历（在 runner 脚本 `gym.make()` 之后）：
```python
_stage = omni.usd.get_context().get_stage()
for _prim in _stage.Traverse():
    if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)
```

两步必须同时存在。双臂任务同理（`"Robot"` 可同时匹配 `Left_Robot` 和 `Right_Robot`）。

### 2. 关节阻尼（仅状态机需要）

IK 控制器需要比机械臂默认阻尼（0.6 N·m·s/rad）更高的阻尼值，才能获得稳定、平滑的轨迹。阻尼设置为 **10.0 N·m·s/rad**，在每步调用：

```python
# 在 PickOrangeStateMachine.get_action() 中：
robot = env.scene["robot"]
robot.write_joint_damping_to_sim(damping=10.0)
```

**回放时也必须应用相同阻尼**，以匹配录制条件。状态机回放脚本（`scripts/environments/state_machine/replay.py`）在每次 `env.step()` 前调用 `apply_damping(env, task_type)`。

> RL 训练无需此设置——直接关节位置控制本身不依赖高阻尼。

### 3. Episode 编号规则

IsaacLab 录制器在第一次 `env.reset()` 调用时（此时还未执行任何步骤）会保存一个仅含初始状态的 episode（`num_samples=0`），即 `demo_0`。

| `--select_episodes N` | 加载的 episode | 内容 |
|---|---|---|
| 0 | `demo_0` | 空（无 actions）—— 会导致 `TypeError` |
| 1 | `demo_1` | 第 1 条真实演示 |
| K | `demo_K` | 第 K 条真实演示 |

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `scripts/environments/state_machine/pick_orange.py` | 拾橙任务录制 Runner |
| `scripts/environments/state_machine/replay.py` | 状态机专用回放脚本（含阻尼设置） |
| `scripts/rl/train_pick_orange.py` | 拾橙任务 PPO RL 训练 |
| `source/leisaac/leisaac/state_machine/base.py` | `StateMachineBase` 抽象基类 |
| `source/leisaac/leisaac/state_machine/pick_orange.py` | `PickOrangeStateMachine` |
| `source/leisaac/leisaac/rl/rsl_rl_wrapper.py` | IsaacLab → rsl_rl VecEnv 包装器 |
| `source/leisaac/leisaac/tasks/pick_orange/pick_orange_rl_env_cfg.py` | RL 环境配置（观测、奖励、动作） |
| `replay_pick_orange.sh` | replay.py 的 Shell 封装脚本 |
| `record_pick_orange.sh` | 拾橙录制的 Shell 封装脚本 |
