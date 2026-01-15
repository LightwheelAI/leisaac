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


def randomize_deformable_object_uniform(
    name: str,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> EventTerm:
    if velocity_range is None:
        velocity_range = {}
    return EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={"position_range": pose_range, "velocity_range": velocity_range, "asset_cfg": SceneEntityCfg(name)},
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


def randomize_mixed_objects_uniform(
    rigid_names: list[str] | None = None,
    deformable_names: list[str] | None = None,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> EventTerm:

    if rigid_names is None:
        rigid_names = []
    if deformable_names is None:
        deformable_names = []
    if pose_range is None:
        pose_range = {}
    if velocity_range is None:
        velocity_range = {}
    return EventTerm(
        func=enhance_mdp.reset_mixed_objects_uniform,
        mode="reset",
        params={
            "pose_range": pose_range,
            "velocity_range": velocity_range,
            "rigid_asset_cfg": [SceneEntityCfg(name) for name in rigid_names],
            "deformable_asset_cfg": [SceneEntityCfg(name) for name in deformable_names],
        },
    )
