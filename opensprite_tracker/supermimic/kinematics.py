"""Differentiable Sprite link-origin FK and exact official box-sole geometry."""
import mujoco
import numpy as np
import torch

from .robot import MJCF_PATH, JOINT_NAMES, TRACKED_BODIES, default_joint_pos
from .rotations import axis_angle_to_mat, quat_to_mat


class OpenSpriteKinematics:
    def __init__(self, device="cpu", tracked=None, model=None, dtype=torch.float32,
                 active_collision_only=False):
        m = model if model is not None else mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.device, self.dtype = str(device), dtype
        self.joint_names = tuple(JOINT_NAMES)
        self.tracked_names = list(TRACKED_BODIES if tracked is None else tracked)
        self.body_names = [m.body(b).name.rsplit("/", 1)[-1] for b in range(1, m.nbody)]
        self.tracked_idx = [self.body_names.index(n) for n in self.tracked_names]
        self.num_bodies = len(self.body_names)
        self.name = "OpenSprite0825"
        self.default_q = default_joint_pos(device, dtype)
        self.defaultq = self.default_q
        tensor = lambda x: torch.as_tensor(np.array(x), device=device, dtype=dtype)
        self.off_p, self.off_R = tensor(m.body_pos[1:]), quat_to_mat(tensor(m.body_quat[1:]))
        self.parent = [int(m.body_parentid[b]) - 1 for b in range(1, m.nbody)]
        self.joint_idx, pivots, axes, qrefs = [], [], [], []
        lo, hi = np.zeros(len(self.joint_names)), np.zeros(len(self.joint_names))
        found = set()
        for b in range(1, m.nbody):
            count, j = int(m.body_jntnum[b]), int(m.body_jntadr[b])
            if count == 0:
                idx, pivot, axis, qref = -2, np.zeros(3), np.zeros(3), 0.
            elif count == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                idx, pivot, axis, qref = -1, np.zeros(3), np.zeros(3), 0.
            elif count == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                name = m.joint(j).name.rsplit("/", 1)[-1]
                idx = self.joint_names.index(name)
                found.add(name)
                pivot, axis, qref = m.jnt_pos[j], m.jnt_axis[j], m.qpos0[m.jnt_qposadr[j]]
                lo[idx], hi[idx] = m.jnt_range[j]
            else:
                raise ValueError("Sprite FK expects fixed bodies, one free root, and single hinges")
            self.joint_idx.append(idx)
            pivots.append(pivot); axes.append(axis); qrefs.append(qref)
        if found != set(self.joint_names):
            raise ValueError("Model joint names do not match OpenSprite0825")
        self.pivots, self.axes, self.qref = tensor(pivots), tensor(axes), tensor(qrefs)
        self.q_lo, self.q_hi = tensor(lo), tensor(hi)
        # Geometry shared with the baseline clearance calculation; no other robot assets.
        self.foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        centers, sole_rotations, halves = [], [], []
        for side in ("left", "right"):
            gid = next(g for g in range(m.ngeom)
                       if m.geom(g).name.rsplit("/", 1)[-1] == f"{side}_foot_collision")
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Sprite requires its official box sole")
            centers.append(m.geom_pos[gid]); halves.append(m.geom_size[gid])
            sole_rotations.append(m.geom_quat[gid])
        self.sole_centers, self.sole_half = tensor(centers), tensor(halves)
        self.sole_R = quat_to_mat(tensor(sole_rotations))
        # The official model has no hardware IMU site. Explicit virtual sensor
        # orientation at the Sprite waist-yaw link, not another robot's sensor.
        self.imu_body_idx = self.body_names.index("waist_yaw_link")
        self.imu_site_R = torch.eye(3, device=device, dtype=dtype)
        heel_to_toe_local = torch.zeros((2, 3), device=device, dtype=dtype)
        heel_to_toe_local[:, 0] = -2*self.sole_half[:, 0]
        self.heel_minus_toe = (self.sole_R @ heel_to_toe_local[..., None]).squeeze(-1)
        self.total_mass_kg = float(m.body_mass[1:].sum())
        # Reference-only support ambiguity check. Even geoms disabled by the
        # feet-only scene remain useful here as robot geometry, not contacts.
        # Near-ground non-foot geometry only means foot-only support is UNKNOWN,
        # never a claim that the reference or actual robot is in contact.
        gids = [g for g in range(m.ngeom)
                if (m.geom(g).name or "").endswith("_collision")
                and (not active_collision_only or m.geom_contype[g] or m.geom_conaffinity[g])
                and m.geom_bodyid[g] > 0
                and m.body(int(m.geom_bodyid[g])).name.rsplit("/", 1)[-1]
                not in self.foot_names]
        self.nonfoot_body_idx = [int(m.geom_bodyid[g])-1 for g in gids]
        self.nonfoot_geom_pos = tensor(m.geom_pos[gids])
        self.nonfoot_geom_R = quat_to_mat(tensor(m.geom_quat[gids]))
        self.nonfoot_types = [int(m.geom_type[g]) for g in gids]
        self.nonfoot_sizes = tensor(m.geom_size[gids])
        self._reference_mesh_vertices = {}
        self._reference_hull_vertices = None  # Lazy: paper profile never needs this.
        for i, g in enumerate(gids):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                adr, num = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                self._reference_mesh_vertices[i] = np.array(m.mesh_vert[adr:adr+num])

    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        P, Rs, V, W = [], [], [], []
        for i, (par, j) in enumerate(zip(self.parent, self.joint_idx)):
            if j == -1:
                pi, Ri, vi, wi = p, R, v, w
            else:
                if par < 0:
                    pp, Rp, vp, wp = torch.zeros_like(p), torch.eye(3, device=p.device, dtype=p.dtype).expand_as(R), torch.zeros_like(v), torch.zeros_like(w)
                else:
                    pp, Rp, vp, wp = P[par], Rs[par], V[par], W[par]
                p0 = pp + (Rp @ self.off_p[i, :, None]).squeeze(-1)
                R0 = Rp @ self.off_R[i]
                if j >= 0:
                    J = axis_angle_to_mat(self.axes[i], q[..., j] - self.qref[i])
                    Ri = R0 @ J
                    pi = p0 + (R0 @ (self.pivots[i] - (J @ self.pivots[i, :, None]).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
                    axis_w = (R0 @ self.axes[i, :, None]).squeeze(-1)
                    dw = axis_w * qd[..., j:j+1]
                    pivot_w = p0 + (R0 @ self.pivots[i, :, None]).squeeze(-1)
                    wi = wp + dw
                    vi = vp + torch.cross(wp, pi - pp, dim=-1) + torch.cross(dw, pi - pivot_w, dim=-1)
                else:
                    pi, Ri, wi = p0, R0, wp
                    vi = vp + torch.cross(wp, pi - pp, dim=-1)
            P.append(pi); Rs.append(Ri); V.append(vi); W.append(wi)
        ids = range(self.num_bodies) if all_bodies else self.tracked_idx
        return tuple(torch.stack([items[i] for i in ids], dim=dim)
                     for items, dim in ((P, -2), (Rs, -3), (V, -2), (W, -2)))

    def imu(self, *state):
        """Return torso IMU orientation (world), gyro (site), gravity (site)."""
        _, R, _, w = self.forward(*state[:6], all_bodies=True)
        sensor_R = R[..., self.imu_body_idx, :, :] @ self.imu_site_R
        gyro = (sensor_R.transpose(-1, -2) @ w[..., self.imu_body_idx, :, None]).squeeze(-1)
        return sensor_R, gyro, -sensor_R[..., 2, :]

    def sole_bottom_z(self, pos, rotation):
        """Exact box support minimum; inputs (...,2,3)/(...,2,3,3)."""
        centers = pos + (rotation @ self.sole_centers[..., None]).squeeze(-1)
        box_R = rotation @ self.sole_R
        return centers[..., 2] - (box_R[..., 2, :].abs()*self.sole_half).sum(-1)

    def sole_normals_world(self, rotation):
        return (rotation @ self.sole_R)[..., :, 2]

    def reference_nonfoot_clearance(self, *state):
        """Minimum non-foot collider height above flat z=0.

        Used only on immutable reference states. No upright/root-height test or
        list of named poses is involved. Use the original collision surfaces,
        not a rotated box enclosing a bent limb (which gives false contacts).
        This does not enable full-body collisions in the physical simulator.
        """
        if self._reference_hull_vertices is None:
            from scipy.spatial import ConvexHull
            # Linear support minima on a mesh equal those on its convex hull.
            # This reduces initialization work/memory without changing height.
            self._reference_hull_vertices = {
                i: torch.as_tensor(vertices[ConvexHull(vertices).vertices],
                                   device=self.device, dtype=self.dtype)
                for i, vertices in self._reference_mesh_vertices.items()}
        p, R, _, _ = self.forward(*state[:6], all_bodies=True)
        gp = p[..., self.nonfoot_body_idx, :] + (
            R[..., self.nonfoot_body_idx, :, :] @ self.nonfoot_geom_pos[..., None]
        ).squeeze(-1)
        gR = R[..., self.nonfoot_body_idx, :, :] @ self.nonfoot_geom_R
        lower = []
        for i, kind in enumerate(self.nonfoot_types):
            z_axis, size = gR[..., i, 2, :], self.nonfoot_sizes[i]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                vertices = self._reference_hull_vertices[i]
                dz = (z_axis @ vertices.T).amin(-1)
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                dz = -(z_axis.abs()*size).sum(-1)
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                dz = -(size[1]*z_axis[..., 2].abs()
                       + size[0]*(z_axis[..., :2].square().sum(-1)).sqrt())
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                dz = -size[0] - size[1]*z_axis[..., 2].abs()
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                dz = z_axis[..., 0]*0-size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                dz = -(z_axis*size).square().sum(-1).sqrt()
            else:
                raise ValueError(f"Unsupported Sprite reference collision geom type {kind}")
            lower.append(gp[..., i, 2] + dz)
        return torch.stack(lower, -1).amin(-1)

    def ground_clearance(self, *state):
        """Lowest actual Sprite primitive/box-sole surface; never a force."""
        nonfoot = self.reference_nonfoot_clearance(*state)
        bp, bR, _, _ = self.forward(*state[:6])
        feet = [self.tracked_names.index(n) for n in self.foot_names]
        p, R = bp[..., feet, :], bR[..., feet, :, :]
        sole = self.sole_bottom_z(p, R).amin(-1)
        return torch.minimum(nonfoot, sole)


Kinematics = OpenSpriteKinematics
