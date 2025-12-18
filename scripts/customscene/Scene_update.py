"""
Scene Composition 

Task types:
- toys: Toyroom toy blocks
- orange: Kitchen orange
- cloth: Bedroom cloth
- cube: Table with cube
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import Transformation_constant as TC
from leisaac.utils.constant import ASSETS_ROOT

# Task configuration
TASK_CONFIG = {
    "toys": {
        "layout": TC.Toyroom_LAYOUT,
        "assets_path": f"{ASSETS_ROOT}/scenes/lightwheel_toyroom/Assets",
        "table_name": "KidRoom_Table01",  
    },
    "orange": {
        "layout": TC.Kitchen_LAYOUT,
        "assets_path": f"{ASSETS_ROOT}/scenes/kitchen_with_orange/objects",
    },
    "cloth": {
        "layout": TC.Bedroom_LAYOUT,
        "assets_path": f"{ASSETS_ROOT}/scenes/lightwheel_bedroom",
        "table_name": "Table038_01",  
    },
    "cube": {
        "layout": TC.Cube_LAYOUT,
        "assets_path": f"{ASSETS_ROOT}/scenes/table_with_cube/cube",
    },
}


# Get USD path
def _get_usd_path(task_type: str, obj_name: str, base: str, config: dict) -> str:

    table_name = config.get("table_name")
    
    # toys
    if task_type == "toys":
        if obj_name == table_name:
            return f"{base}/{obj_name}/{obj_name}.usd"
        name = obj_name[:-3] if obj_name.endswith("_01") else obj_name
        return f"{base}/Kit1/{name}.usd"
    
    # orange
    if task_type == "orange":
        return f"{base}/{obj_name}/{obj_name}.usd"
    
    # cloth
    if task_type == "cloth":
        if obj_name == table_name:
            folder_name = obj_name[:-3] if obj_name.endswith("_01") else obj_name
            return f"{base}/LW_Loft/Loft/{folder_name}/{folder_name}.usd"
        if obj_name == "cloth":
            return f"{base}/cloth/cloth.usd"
        raise ValueError(f"Unknown cloth object: {obj_name}")
    
    # cube
    if task_type == "cube":
        if obj_name == "cube":
            return f"{base}/cube.usd"
        raise ValueError(f"Unknown cube object: {obj_name}")
    
    raise ValueError(f"Unknown task type: {task_type}")


# Transformation Functions

def _quat_to_rot(q):

    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n > 0:
        w, x, y, z = w / n, x / n, y / n, z / n

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _rot_to_quat(R):

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        return [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        return [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]


def _pose_to_matrix(pos, quat):

    M = np.eye(4)
    M[:3, :3] = _quat_to_rot(quat)
    M[:3, 3] = pos
    return M


def _matrix_to_pose(M):
    
    return M[:3, 3].tolist(), _rot_to_quat(M[:3, :3])


def transform_layout(
    layout: Dict[str, Dict[str, list]],
    orig_pos: Union[Tuple, List],
    orig_quat: Union[Tuple, List],
    target_pos: Union[Tuple, List],
    target_quat: Union[Tuple, List],
) -> Dict[str, Dict[str, List[float]]]:
    """Transform all object poses from original to target reference frame"""
    M_orig = _pose_to_matrix(orig_pos, orig_quat)
    M_target = _pose_to_matrix(target_pos, target_quat)
    T = M_target @ np.linalg.inv(M_orig)

    result = {}
    for name, data in layout.items():
        M = T @ _pose_to_matrix(data["pos"], data["rot"])
        pos, quat = _matrix_to_pose(M)
        result[name] = {"pos": pos, "rot": quat}

    return result


# Layout Access

def get_task_types() -> List[str]:
    """Return all supported task types"""
    return list(TASK_CONFIG.keys())


def get_layout(task_type: str) -> Dict[str, Dict[str, list]]:
    """Retrieve the layout dictionary for a given task type"""
    if task_type not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_type}. Available: {get_task_types()}")
    return TASK_CONFIG[task_type]["layout"]


# Scene Creation

def compose_scene(
    task_type: str,
    background_usd: str,
    output_usd: str,
    orig_pos: Union[Tuple, List],
    orig_quat: Union[Tuple, List],
    target_pos: Union[Tuple, List],
    target_quat: Union[Tuple, List],
    include_table: bool = False,
) -> str:
    """
    Compose a USD scene by placing task objects into a background scene
    """
    # Import USD modules here (after Isaac Sim initialization)
    from pxr import Usd, UsdGeom, Gf
    
    if task_type not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_type}. Available: {get_task_types()}")

    config = TASK_CONFIG[task_type]
    assets_base = config["assets_path"]
    layout = config["layout"]

    # Check if table is supported for this task
    table_name = config.get("table_name")
    if include_table and not table_name:
        print(f"[WARN] Table reference not supported for '{task_type}', ignoring include_table flag")
        include_table = False

    # Filter layout: exclude table if not include_table
    filtered_layout = {k: v for k, v in layout.items() if include_table or k != table_name}
    
    transformed = transform_layout(filtered_layout, orig_pos, orig_quat, target_pos, target_quat)

    stage = Usd.Stage.CreateNew(output_usd)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene_prim = stage.DefinePrim("/World/Scene")
    scene_prim.GetReferences().AddReference(background_usd)

    # Add objects
    for obj_name, pose in transformed.items():
        obj_usd = _get_usd_path(task_type, obj_name, assets_base, config)

        if not Path(obj_usd).exists():
            print(f"[WARN] {obj_usd} not found, skipping {obj_name}")
            continue

        obj_path = f"/World/Scene/{obj_name}"
        obj_prim = stage.DefinePrim(obj_path)
        obj_prim.GetReferences().AddReference(obj_usd)

        xform = UsdGeom.Xformable(obj_prim)
        # Clear existing xform ops to avoid conflicts
        xform.ClearXformOpOrder()
        # Add new transform
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose["pos"]))
        xform.AddOrientOp().Set(Gf.Quatf(*pose["rot"]))

        print(f"[OK] Added {obj_name}")

    stage.Save()
    print(f"[OK] Scene saved: {output_usd}")
    return output_usd


# JSON Export

def transform_to_json(
    task_type: str,
    orig_pos: Union[Tuple, List],
    orig_quat: Union[Tuple, List],
    target_pos: Union[Tuple, List],
    target_quat: Union[Tuple, List],
    include_table: bool = False,
    robot_orig_pos: Union[Tuple, List] = None,
    robot_orig_quat: Union[Tuple, List] = None,
    output_path: str = None,
) -> str:
    """
        JSON string with transformed poses
    """
    if task_type not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_type}. Available: {get_task_types()}")
    
    config = TASK_CONFIG[task_type]
    layout = config["layout"]
    table_name = config.get("table_name")
    
    # Filter layout: exclude table if not include_table
    filtered_layout = {k: v for k, v in layout.items() if include_table or k != table_name}
    
    # Transform all objects
    result = transform_layout(filtered_layout, orig_pos, orig_quat, target_pos, target_quat)
    
    # Add robot info
    if include_table:
        # When using table reference, transform the robot position too
        robot_layout = {"robot": {"pos": list(robot_orig_pos), "rot": list(robot_orig_quat)}}
        robot_transformed = transform_layout(robot_layout, orig_pos, orig_quat, target_pos, target_quat)
        result["robot"] = robot_transformed["robot"]
    else:
        # When using arm reference, robot is at target position
        result["robot"] = {"pos": list(target_pos), "rot": list(target_quat)}

    json_str = json.dumps(result, indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(json_str)
        print(f"[OK] JSON saved: {output_path}")

    return json_str