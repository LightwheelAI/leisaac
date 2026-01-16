# Available Policy Inference

This page lists the policy inference methods currently supported by LeIsaac.

Depending on your use case, you may need to install additional dependencies to enable inference:

:::tip
For each supported policy, we have specified the verified commit. If the corresponding repository is updated, it may cause compatibility issues. If you encounter such cases, feel free to open an issue.
:::

## Finetuned gr00t n1.5

Install additional dependencies:

```shell
pip install -e "source/leisaac[gr00t]"
```

Inference Scripts:

```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.5 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

:::tip
target commit: https://github.com/NVIDIA/Isaac-GR00T/commit/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
:::

## Finetuned gr00t n1.6

Install additional dependencies:

```shell
pip install -e "source/leisaac[gr00t]"
```

Inference Scripts:

```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.6 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

:::tip
target commit: https://github.com/NVIDIA/Isaac-GR00T/commit/e8e625f4f21898c506a1d8f7d20a289c97a52acf
:::

## Lerobot official policy

Install additional dependencies:

```shell
pip install -e "source/leisaac[lerobot-async]"
```

We utilize lerobot's async inference capabilities for policy execution. For detailed information, please refer to the [official documentation](https://huggingface.co/docs/lerobot/async). Prior to execution, ensure that the policy server is running.


```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras
```

:::tip
target commit: https://github.com/huggingface/lerobot/tree/v0.3.3
:::

## Finetuned openpi

Install additional dependencies:

```shell
pip install -e "source/leisaac[openpi]"
```

We utilize openpi's remote inference capabilities for policy execution. For detailed information, please refer to the [official documentation](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md). Prior to execution, ensure that the policy server is running.

```shell
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=openpi \
    --policy_host=localhost \
    --policy_port=8000 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --device=cuda \
    --enable_cameras
```

:::tip
target commit: https://github.com/Physical-Intelligence/openpi/commit/5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d
:::
