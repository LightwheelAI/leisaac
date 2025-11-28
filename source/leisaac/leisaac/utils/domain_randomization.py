from typing import Literal

import isaaclab.envs.mdp as mdp
import leisaac.enhance.envs.mdp as enhance_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg


def randomize_object_uniform(
    name: str,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> EventTerm:
    if velocity_range is None:
        velocity_range = {}
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": pose_range, "velocity_range": velocity_range, "asset_cfg": SceneEntityCfg(name)},
    )


def randomize_camera_uniform(
    name: str, pose_range: dict[str, tuple[float, float]], convention: Literal["ros", "opengl", "world"] = "ros"
) -> EventTerm:
    return EventTerm(
        func=enhance_mdp.randomize_camera_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "pose_range": pose_range,
            "convention": convention,
        },
    )


def randomize_particle_object_uniform(
    name: str,
    pose_range: dict[str, tuple[float, float]],
) -> EventTerm:
    return EventTerm(
        func=enhance_mdp.randomize_particle_object_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "pose_range": pose_range,
        },
    )


def domain_randomization(env_cfg, random_options: list[EventTerm]):
    for idx, event_item in enumerate(random_options):
        setattr(env_cfg.events, f"domain_randomize_{idx}", event_item)
