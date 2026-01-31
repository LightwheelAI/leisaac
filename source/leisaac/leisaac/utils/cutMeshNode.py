import os
import platform
from dataclasses import dataclass

import numpy as np
import omni
import pip
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


class cutMeshNode:

    @dataclass
    class InnerState:
        _meshPath: str = "null path"

    @staticmethod
    def internal_state():
        return cutMeshNode.InnerState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""
        mesh_path = db.inputs.cut_mesh_path
        # state = db.internal_state
        stage = omni.usd.get_context().get_stage()

        if db.inputs.cutEventIn:

            knife_path = db.inputs.knife_mesh_path
            knife_prim = stage.GetPrimAtPath(knife_path)
            plane_center, plane_normal = get_knife_plane_in_world(knife_prim)

            mesh_prim = stage.GetPrimAtPath(mesh_path)
            for mesh_child in mesh_prim.GetChildren():
                if mesh_child is not None:
                    plane_center_in_prim = compute_world_position_in_prim(plane_center, mesh_child)
                    plane_normal_in_prim = compute_world_dir_in_prim(plane_normal, mesh_child)
                    cut_prim(stage, mesh_child, plane_center_in_prim, plane_normal_in_prim)

        return True


def cut_prim(stage, prim, plane_origin, plane_normal):
    try:
        import trimesh
    except Exception as error:
        print("errrrrrrro")
        package_list = ("trimesh", "shapely", "rtree", "triangle")  # ,
        for package_name in package_list:
            install_package(package_name)
        raise RuntimeError("Find package failed") from error

    def get_Trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get())
        face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        faces = []
        index = 0
        for count in face_vertex_counts:
            if count == 3:
                faces.append(face_vertex_indices[index : index + 3])
            elif count > 3:
                pass
            index += count
        trimesh_mesh = trimesh.Trimesh(points, np.array(faces))
        return trimesh_mesh

    def generate_mesh(meshPrim: UsdGeom.Mesh, trimesh: trimesh.Trimesh):

        new_points = [Gf.Vec3f(*v) for v in trimesh.vertices]
        face_vertex_counts = [3] * len(trimesh.faces)
        face_vertex_indices = trimesh.faces.flatten().tolist()

        meshPrim.CreatePointsAttr(new_points)
        meshPrim.CreateFaceVertexCountsAttr(face_vertex_counts)
        meshPrim.CreateFaceVertexIndicesAttr(face_vertex_indices)

    trimesh_mesh = get_Trimesh(prim)

    plane_normal_opposite = [-plane_normal[0], -plane_normal[1], -plane_normal[2]]
    parent_prim = prim.GetParent()
    cut_mesh_name = prim.GetName()

    # try:
    sub_trimesh1 = trimesh_mesh.slice_plane(plane_origin, plane_normal, cap=True)
    sub_trimesh2 = trimesh_mesh.slice_plane(plane_origin, plane_normal_opposite, cap=True)

    if len(sub_trimesh2.vertices) == 0 or len(sub_trimesh1.vertices) == 0:
        return True

    new_prim_name1 = cut_mesh_name + "_sub1"
    new_prim_path1 = parent_prim.GetPath().AppendChild(new_prim_name1)
    new_mesh1 = UsdGeom.Mesh.Define(stage, new_prim_path1)
    generate_mesh(new_mesh1, sub_trimesh1)
    new_mesh_prim1 = stage.GetPrimAtPath(new_prim_path1)
    if prim.HasAPI(PhysxSchema.PhysxDeformableAPI):
        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=new_mesh_prim1,
            component="PhysxDeformableBodyAPI",
        )
        omni.kit.commands.execute("AddDeformableBodyComponent", skin_mesh_path=new_mesh_prim1.GetPath())
        copy_deformable_properties(prim, new_mesh_prim1)
        copy_transform(prim, new_mesh_prim1)
        copy_material(prim, new_mesh_prim1)

    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=new_mesh_prim1,
            component="PhysicsRigidBodyAPI",
        )

        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=new_mesh_prim1,
            component="PhysicsCollisionAPI",
        )
        # To copy rigid body properties
        copy_transform(prim, new_mesh_prim1)

    new_prim_name2 = cut_mesh_name + "_sub2"
    new_prim_path2 = parent_prim.GetPath().AppendChild(new_prim_name2)
    new_mesh2 = UsdGeom.Mesh.Define(stage, new_prim_path2)
    generate_mesh(new_mesh2, sub_trimesh2)
    new_mesh_prim2 = stage.GetPrimAtPath(new_prim_path2)
    if prim.HasAPI(PhysxSchema.PhysxDeformableAPI):
        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=new_mesh_prim2,
            component="PhysxDeformableBodyAPI",
        )
        omni.kit.commands.execute("AddDeformableBodyComponent", skin_mesh_path=new_mesh_prim2.GetPath())
        copy_deformable_properties(prim, new_mesh_prim2)
        copy_transform(prim, new_mesh_prim2)
        copy_material(prim, new_mesh_prim2)

    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=new_mesh_prim2,
            component="PhysicsRigidBodyAPI",
        )

        omni.kit.commands.execute(
            "AddPhysicsComponent",
            usd_prim=new_mesh_prim2,
            component="PhysicsCollisionAPI",
        )
        # To copy rigid body properties
        copy_transform(prim, new_mesh_prim2)
    prim.SetActive(False)
    # stage.RemovePrim(prim.GetPath())
    # except Exception as error:
    #     raise RuntimeError(f"Cut Mesh failed{prim.GetName()}") from error
    return None


def copy_deformable_properties(primA, primB):
    deformableStateA = PhysxSchema.PhysxDeformableAPI(primA)
    deformableStateB = PhysxSchema.PhysxDeformableAPI(primB)

    deformableEnabled = deformableStateA.GetDeformableEnabledAttr().Get()
    deformableStateB.GetDeformableEnabledAttr().Set(deformableEnabled)

    enableCCD = deformableStateA.GetEnableCCDAttr().Get()
    deformableStateB.GetEnableCCDAttr().Set(enableCCD)

    maxDepenetrationVelocity = deformableStateA.GetMaxDepenetrationVelocityAttr().Get()
    deformableStateB.GetMaxDepenetrationVelocityAttr().Set(maxDepenetrationVelocity)

    velocityDamping = deformableStateA.GetVertexVelocityDampingAttr().Get()
    deformableStateB.GetVertexVelocityDampingAttr().Set(velocityDamping)

    selfCollision = deformableStateA.GetSelfCollisionAttr().Get()
    deformableStateB.GetSelfCollisionAttr().Set(selfCollision)

    selfCollisionFilterDistance = deformableStateA.GetSelfCollisionFilterDistanceAttr().Get()
    deformableStateB.GetSelfCollisionFilterDistanceAttr().Set(selfCollisionFilterDistance)

    settlingThreshold = deformableStateA.GetSettlingThresholdAttr().Get()
    deformableStateB.GetSettlingThresholdAttr().Set(settlingThreshold)

    sleepDamping = deformableStateA.GetSleepDampingAttr().Get()
    deformableStateB.GetSleepDampingAttr().Set(sleepDamping)

    sleepThreshold = deformableStateA.GetSleepThresholdAttr().Get()
    deformableStateB.GetSleepThresholdAttr().Set(sleepThreshold)

    solverPositionIteration = deformableStateA.GetSolverPositionIterationCountAttr().Get()
    deformableStateB.GetSolverPositionIterationCountAttr().Set(solverPositionIteration)

    collisionStateA = PhysxSchema.PhysxCollisionAPI(primA)
    collisionStateB = PhysxSchema.PhysxCollisionAPI(primB)

    contactOffset = collisionStateA.GetContactOffsetAttr().Get()
    collisionStateB.GetContactOffsetAttr().Set(contactOffset)

    restOffset = collisionStateA.GetRestOffsetAttr().Get()
    collisionStateB.GetRestOffsetAttr().Set(restOffset)

    if primA.HasAttribute("physxDeformable:simulationHexahedralResolution"):
        value = primA.GetAttribute("physxDeformable:simulationHexahedralResolution").Get()
        primB.GetAttribute("physxDeformable:simulationHexahedralResolution").Set(value)


def copy_transform(primA, primB):
    xformableA = UsdGeom.Xformable(primA)
    ops = xformableA.GetOrderedXformOps()

    xformableB = UsdGeom.Xformable(primB)
    xformableB.ClearXformOpOrder()

    for op in ops:
        new_op = xformableB.AddXformOp(op.GetOpType(), op.GetPrecision())
        new_op.Set(op.Get())


def copy_material(primA, primB):
    from pxr import UsdShade

    looks_material_path = UsdShade.MaterialBindingAPI(primA).GetDirectBindingRel("").GetTargets()[0]
    omni.kit.commands.execute(
        "BindMaterial",
        material_path=str(looks_material_path),
        prim_path=[primB.GetPath()],
        strength=["weakerThanDescendants"],
        material_purpose="",
    )

    physics_material_path = UsdShade.MaterialBindingAPI(primA).GetDirectBindingRel("physics").GetTargets()[0]
    omni.kit.commands.execute(
        "BindMaterial",
        material_path=str(physics_material_path),
        prim_path=[primB.GetPath()],
        strength=["weakerThanDescendants"],
        material_purpose="physics",
    )


def compute_A_position_in_B_space(position, body0_prim, body1_prim):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    usd_current_time = Usd.TimeCode(current_time)
    body1_xform = UsdGeom.Xformable(body1_prim)
    body0_xform = UsdGeom.Xformable(body0_prim)

    Trans0 = body0_xform.ComputeLocalToWorldTransform(usd_current_time)
    p0_in_world = Trans0.Transform(position)

    Trans1 = body1_xform.ComputeLocalToWorldTransform(usd_current_time)
    p0_in_p1 = Trans1.GetInverse().Transform(p0_in_world)

    return p0_in_p1


def compute_prim_position_in_world(position, prim: Usd.Prim):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    usd_current_time = Usd.TimeCode(current_time)

    prim_xform = UsdGeom.Xformable(prim)
    Trans0 = prim_xform.ComputeLocalToWorldTransform(usd_current_time)
    position_in_world = Trans0.Transform(position)

    return position_in_world


def get_knife_plane_in_world(knife_prim):
    #  p6 p7
    # p4 p5
    #   p2 p3
    # p0 p1
    knife_cube = UsdGeom.Cube(knife_prim)
    extent = knife_cube.GetExtentAttr().Get()
    center = (extent[0] + extent[1]) / 2

    p0 = extent[0]
    p7 = extent[1]
    p3 = Gf.Vec3f(extent[1][0], extent[1][1], extent[0][0])
    p1 = Gf.Vec3f(p3[0], extent[0][1], extent[0][2])

    center_in_world = compute_prim_position_in_world(center, knife_prim)
    p0_in_world = compute_prim_position_in_world(p0, knife_prim)
    p7_in_world = compute_prim_position_in_world(p7, knife_prim)
    p3_in_world = compute_prim_position_in_world(p3, knife_prim)
    p1_in_world = compute_prim_position_in_world(p1, knife_prim)

    length13 = (p1_in_world - p3_in_world).GetLength()
    length01 = (p0_in_world - p1_in_world).GetLength()
    length37 = (p3_in_world - p7_in_world).GetLength()

    knife_normal = None
    if length13 < length01 and length13 < length37:
        knife_normal = p1_in_world - p3_in_world
    elif length01 < length13 and length01 < length37:
        knife_normal = p0_in_world - p1_in_world
    elif length37 < length13 and length37 < length01:
        knife_normal = p3_in_world - p7_in_world

    return center_in_world, knife_normal.GetNormalized()


def compute_world_position_in_prim(wolrd_position, prim: Usd.Prim):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    usd_current_time = Usd.TimeCode(current_time)

    prim_xform = UsdGeom.Xformable(prim)
    Trans0 = prim_xform.ComputeLocalToWorldTransform(usd_current_time)
    position_in_prim = Trans0.GetInverse().Transform(wolrd_position)

    return position_in_prim


def compute_world_dir_in_prim(wolrd_dir, prim: Usd.Prim):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    usd_current_time = Usd.TimeCode(current_time)

    prim_xform = UsdGeom.Xformable(prim)
    Trans0 = prim_xform.ComputeLocalToWorldTransform(usd_current_time)
    position_in_prim = Trans0.GetInverse().TransformDir(wolrd_dir)

    return position_in_prim.GetNormalized()


def install_package(package_name, install_name=None):
    """
    Installs a package from a local wheel file or Ali Cloud image.
    Parameters:
    package_name: The name of the package to import
    install_name: The name of the package to install (if different from the import name)
    """
    if install_name is None:
        install_name = package_name

    print(f"{package_name} module not found. Trying to install...", flush=True)
    try:
        # Get the current operating system
        current_os = platform.system()

        # Local wheel file path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        whls_dir = os.path.join(current_dir, "whls")

        # Find the wheel file in the whls directory
        package_whl = None
        if os.path.exists(whls_dir):
            for file in os.listdir(whls_dir):
                if file.startswith(install_name) and file.endswith(".whl"):
                    # Filter wheel files based on operating system
                    is_windows_match = current_os == "Windows" and (
                        "win" in file.lower() or "py3-none-any" in file.lower()
                    )
                    is_linux_match = current_os == "Linux" and (
                        "linux" in file.lower() or "manylinux" in file.lower() or "py3-none-any" in file.lower()
                    )
                    if is_windows_match or is_linux_match:
                        package_whl = os.path.join(whls_dir, file)
                        break

        if package_whl and os.path.exists(package_whl):
            # Install from a local wheel file
            print(
                f"Install from a local wheel file{install_name}: {package_whl}",
                flush=True,
            )
            pip.main(["install", package_whl])
        else:
            # If the wheel file is not found, install it from the Alibaba Cloud image.
            print(
                f"No local wheel file found for {current_os}. Installing {install_name} from Alibaba Cloud image...",
                flush=True,
            )
            pip.main([
                "install",
                install_name,
                "-i",
                "https://mirrors.aliyun.com/pypi/simple/",
                "--trusted-host",
                "mirrors.aliyun.com",
            ])

        print(f"{install_name} was successfully installed", flush=True)
        return True
    except Exception as e:
        print(f"Installation of {install_name} failed: {str(e)}", flush=True)
        return False
