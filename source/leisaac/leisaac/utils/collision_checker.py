import numpy as np
import omni
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom


class Collision_Checker:
    """
    Real-time collision detection package
    See usage example at the bottom
    """

    def __init__(
        self,
        stage=None,
        prim_path0="/World/Scene/Sausage001/Trigger/Cube",
        prim_path1="/World/Robot/Right_Robot/gripper/Knife/Knife/Knife002",
        apply_world_transform=True,
        debug=False,
    ):
        """
        Parameters:
        - stage: pxr Usd stage (defaults to automatically getting the current stage)
        - dynamic_vertices: If True, re-read the prim's vertices for each check (for meshes with vertex deformations)
        - enable_proximity: If True, attempts to return the closest point and distance on collision (may be slow)
        """
        self.stage = stage
        self.apply_world_transform = apply_world_transform
        self.prim_path0 = prim_path0
        self.prim_path1 = prim_path1
        self.debug = debug

    def get_current_timecode(self):
        """
        Returns the Usd.TimeCode corresponding to the current time (usually in frames)
        """
        timeline = omni.timeline.get_timeline_interface()
        current_time = timeline.get_current_time()
        return Usd.TimeCode(current_time)

    def from_str2Usd_Prim(self, prim_path):
        path = Sdf.Path(prim_path)
        prim = self.stage.GetPrimAtPath(path)
        return prim

    def usd_mesh_to_trimesh(self, prim_path):
        prim = self.from_str2Usd_Prim(prim_path)
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)  # (N,3)
        face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        faces = []
        index = 0
        for count in face_vertex_counts:
            if count == 3:
                faces.append(face_vertex_indices[index : index + 3])
            elif count > 3:
                # fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
                base = face_vertex_indices[index]  # first vertex index
                for i in range(1, count - 1):
                    faces.append([
                        base,
                        face_vertex_indices[index + i],
                        face_vertex_indices[index + i + 1],
                    ])
            else:
                # count < 3: Ignore or report an error (usually should not occur)
                # Choose to ignore:
                pass
            index += count

        trimesh_mesh = trimesh.Trimesh(points, np.array(faces))
        return trimesh_mesh

    def compute_A_position_in_B_space(self, position, primA, primB):
        timeline = omni.timeline.get_timeline_interface()
        current_time = timeline.get_current_time()
        usd_current_time = Usd.TimeCode(current_time)

        body0_xform = UsdGeom.Xformable(primA)
        body1_xform = UsdGeom.Xformable(primB)

        Trans0 = body0_xform.ComputeLocalToWorldTransform(usd_current_time)
        p0_in_world = Trans0.Transform(position)

        Trans1 = body1_xform.ComputeLocalToWorldTransform(usd_current_time)
        p0_in_p1 = Trans1.GetInverse().Transform(p0_in_world)
        return p0_in_p1

    def transfer_A_trimesh_in_B_space(self, mesh: trimesh.Trimesh, primA: str, primB: str):
        primA = self.from_str2Usd_Prim(primA)
        primB = self.from_str2Usd_Prim(primB)
        new_vertices = []
        for vertex in mesh.vertices:
            tmp_vertex = self.compute_A_position_in_B_space(Gf.Vec3d(vertex[0], vertex[1], vertex[2]), primA, primB)
            vertex = [tmp_vertex[0], tmp_vertex[1], tmp_vertex[2]]
            new_vertices.append(vertex)
        mesh.vertices = np.array(new_vertices)
        return mesh

    def world_aabb_from_world_trimesh(self, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a local-space mesh (trimesh) and a Local->World matrix mat, compute the world-space AABB.
        Only transform the matrix at the 8 corner points (efficient and accurate).
        """
        local_min, local_max = mesh.bounds  # (3,), (3,)
        corners = np.array(
            [
                [local_min[0], local_min[1], local_min[2]],
                [local_min[0], local_min[1], local_max[2]],
                [local_min[0], local_max[1], local_min[2]],
                [local_min[0], local_max[1], local_max[2]],
                [local_max[0], local_min[1], local_min[2]],
                [local_max[0], local_min[1], local_max[2]],
                [local_max[0], local_max[1], local_min[2]],
                [local_max[0], local_max[1], local_max[2]],
            ],
            dtype=np.float64,
        )
        h = np.ones((8, 4), dtype=np.float64)
        h[:, :3] = corners
        transformed = h
        transformed_xyz = transformed[:, :3] / transformed[:, 3:4]
        world_min = transformed_xyz.min(axis=0)
        world_max = transformed_xyz.max(axis=0)
        if self.debug:
            print(" corners local:", corners)
            print(" corners world:", transformed_xyz)
            print(" world_min, world_max:", world_min, world_max)
        return world_min, world_max

    def aabb_overlap(self, min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> bool:
        return np.all(max_a >= min_b) and np.all(max_b >= min_a)

    def meshes_aabb_collide(self):
        mesh_a = self.usd_mesh_to_trimesh(self.prim_path0)
        mesh_b = self.usd_mesh_to_trimesh(self.prim_path1)
        mesh_a_world = self.transfer_A_trimesh_in_B_space(mesh_a, self.prim_path0, "/World")
        mesh_b_world = self.transfer_A_trimesh_in_B_space(mesh_b, self.prim_path1, "/World")
        min_a, max_a = self.world_aabb_from_world_trimesh(mesh_a_world)
        min_b, max_b = self.world_aabb_from_world_trimesh(mesh_b_world)
        collides = self.aabb_overlap(min_a, max_a, min_b, max_b)
        if self.debug:
            print("A AABB:", min_a, max_a)
            print("B AABB:", min_b, max_b)
            print("collides:", collides)
        return collides, (min_a, max_a), (min_b, max_b)
