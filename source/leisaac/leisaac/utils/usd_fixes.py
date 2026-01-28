# utils/usd_fixes.py
"""
USD-related utilities (best-effort fixes).
"""
import omni.usd
from pxr import UsdGeom, UsdPhysics

def auto_fix_collision_issues():
    """
    Convert certain mesh collisions to convex decomposition to reduce contact issues.
    Best-effort: requires a valid current stage.
    """
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return

    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if any(k in name for k in ("handle", "drawer", "board")):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            api = UsdPhysics.MeshCollisionAPI(prim)
            if not api:
                api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            approx_attr = api.GetApproximationAttr()
            try:
                current_val = approx_attr.Get()
            except Exception:
                current_val = None
            if current_val != "convexDecomposition":
                approx_attr.Set("convexDecomposition")
