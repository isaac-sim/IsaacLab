"""Diagnose DisplayPort plug/socket insertion geometry.

Analyzes the fixed USD assets to determine:
  1. Whether the plug connector cross-section fits inside the socket cavity
  2. The correct world-space insertion pose (position + quaternion)

Usage:
    /path/to/isaaclab/python scripts/diagnose_dp_insertion.py
"""

from __future__ import annotations

import math
import os

from pxr import Gf, Usd, UsdGeom

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "source",
    "isaaclab_tasks",
    "isaaclab_tasks",
    "manager_based",
    "manipulation",
    "deploy",
    "cable_insertion",
    "display_cable_insertion_assets",
)

PLUG_PATH = os.path.join(ASSETS_DIR, "display_port_plug_fixed.usd")
SOCKET_PATH = os.path.join(ASSETS_DIR, "display_port_socket_fixed.usd")

# Socket rotation used in the env config (x, y, z, w)
SOCKET_ROT_XYZW = (0.5, 0.5, 0.5, -0.5)
SOCKET_POS = (0.0, 0.0, 0.15)

PLUG_ROT_XYZW = (0.70711, 0.70711, 0.0, 0.0)


def _load_points(usd_path: str) -> list[Gf.Vec3f]:
    """Load all mesh points from a single-mesh fixed USD."""
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Mesh):
            return list(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    raise RuntimeError(f"No mesh found in {usd_path}")


def _load_body_points_original(usd_path: str, scale: float) -> dict[str, list[Gf.Vec3d]]:
    """Load per-body world-space points from the ORIGINAL (pre-fix) USD."""
    stage = Usd.Stage.Open(usd_path)
    xf_cache = UsdGeom.XformCache()
    bodies: dict[str, list[Gf.Vec3d]] = {}
    seen_keys: set[tuple[str, int]] = set()
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        parent = prim.GetParent().GetName()
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        key = (parent, len(pts))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        world_xf = xf_cache.GetLocalToWorldTransform(prim)
        world_pts = []
        for p in pts:
            wp = world_xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            world_pts.append(Gf.Vec3d(wp[0] * scale, wp[1] * scale, wp[2] * scale))
        bodies[parent] = world_pts
    return bodies


def _bbox(pts):
    """Return (min_xyz, max_xyz) for a list of points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _quat_to_matrix(q_xyzw):
    """Convert (x,y,z,w) quaternion to a Gf.Matrix4d rotation."""
    x, y, z, w = q_xyzw
    rot = Gf.Rotation(Gf.Quatd(w, x, y, z))
    m = Gf.Matrix4d()
    m.SetRotate(rot)
    return m


def _quat_mul(q1, q2):
    """Multiply two (x,y,z,w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def _quat_inv(q):
    """Invert a unit quaternion (x,y,z,w)."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def _rotate_vec(q_xyzw, v):
    """Rotate vector v by quaternion q (x,y,z,w)."""
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = v
    tx = 2.0*(qy*vz - qz*vy)
    ty = 2.0*(qz*vx - qx*vz)
    tz = 2.0*(qx*vy - qy*vx)
    return (
        vx + qw*tx + qy*tz - qz*ty,
        vy + qw*ty + qz*tx - qx*tz,
        vz + qw*tz + qx*ty - qy*tx,
    )


def _transform_points_by_pose(pts, pos, rot_xyzw):
    """Apply rigid body transform (pos, rot) to a list of points."""
    m = _quat_to_matrix(rot_xyzw)
    out = []
    for p in pts:
        rp = m.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        out.append(Gf.Vec3d(rp[0]+pos[0], rp[1]+pos[1], rp[2]+pos[2]))
    return out


def _cross_section_at_x(pts, x_val, tol=0.0005):
    """Get Y-Z bounds of points near a given X value."""
    nearby = [(p[1], p[2]) for p in pts if abs(p[0] - x_val) < tol]
    if not nearby:
        return None
    ys = [p[0] for p in nearby]
    zs = [p[1] for p in nearby]
    return (min(ys), max(ys)), (min(zs), max(zs))


def main():
    print("=" * 70)
    print("DisplayPort Insertion Geometry Diagnostic")
    print("=" * 70)

    # --- 1. Load fixed assets and show local-frame bounding boxes ---
    plug_pts = _load_points(PLUG_PATH)
    socket_pts = _load_points(SOCKET_PATH)

    plug_min, plug_max = _bbox(plug_pts)
    sock_min, sock_max = _bbox(socket_pts)

    print("\n--- Local-frame bounding boxes (mm) ---")
    print(f"Plug:   X[{plug_min[0]*1e3:.2f}, {plug_max[0]*1e3:.2f}]  "
          f"Y[{plug_min[1]*1e3:.2f}, {plug_max[1]*1e3:.2f}]  "
          f"Z[{plug_min[2]*1e3:.2f}, {plug_max[2]*1e3:.2f}]")
    print(f"Socket: X[{sock_min[0]*1e3:.2f}, {sock_max[0]*1e3:.2f}]  "
          f"Y[{sock_min[1]*1e3:.2f}, {sock_max[1]*1e3:.2f}]  "
          f"Z[{sock_min[2]*1e3:.2f}, {sock_max[2]*1e3:.2f}]")

    # --- 2. Load original per-body geometry for anatomy ---
    orig_plug_dir = os.path.join(ASSETS_DIR, "display_port_plug.usd")
    orig_sock_dir = os.path.join(ASSETS_DIR, "display_port_socket.usd")

    print("\n--- Per-body anatomy (world-space, mm) ---")
    for label, path, scale in [("PLUG", orig_plug_dir, 0.01), ("SOCKET", orig_sock_dir, 0.01)]:
        bodies = _load_body_points_original(path, scale)
        print(f"\n  {label}:")
        for name, pts in bodies.items():
            bmin, bmax = _bbox(pts)
            print(f"    {name:8s}  X[{bmin[0]*1e3:7.2f},{bmax[0]*1e3:7.2f}]  "
                  f"Y[{bmin[1]*1e3:7.2f},{bmax[1]*1e3:7.2f}]  "
                  f"Z[{bmin[2]*1e3:7.2f},{bmax[2]*1e3:7.2f}]  "
                  f"size({(bmax[0]-bmin[0])*1e3:.2f} x {(bmax[1]-bmin[1])*1e3:.2f} x {(bmax[2]-bmin[2])*1e3:.2f})")

    # --- 3. World-space transforms ---
    print("\n--- World-space bounding boxes (mm) ---")
    sock_world = _transform_points_by_pose(socket_pts, SOCKET_POS, SOCKET_ROT_XYZW)
    sw_min, sw_max = _bbox(sock_world)
    print(f"Socket (world): X[{sw_min[0]*1e3:.2f}, {sw_max[0]*1e3:.2f}]  "
          f"Y[{sw_min[1]*1e3:.2f}, {sw_max[1]*1e3:.2f}]  "
          f"Z[{sw_min[2]*1e3:.2f}, {sw_max[2]*1e3:.2f}]")

    plug_world = _transform_points_by_pose(plug_pts, (0, 0, 0.25), PLUG_ROT_XYZW)
    pw_min, pw_max = _bbox(plug_world)
    print(f"Plug   (world): X[{pw_min[0]*1e3:.2f}, {pw_max[0]*1e3:.2f}]  "
          f"Y[{pw_min[1]*1e3:.2f}, {pw_max[1]*1e3:.2f}]  "
          f"Z[{pw_min[2]*1e3:.2f}, {pw_max[2]*1e3:.2f}]")

    # --- 4. Determine socket cavity axes in world space ---
    # The socket's local X axis is the insertion depth axis.
    # With socket rotation, find where it maps in world space.
    socket_x_world = _rotate_vec(SOCKET_ROT_XYZW, (1, 0, 0))
    socket_y_world = _rotate_vec(SOCKET_ROT_XYZW, (0, 1, 0))
    socket_z_world = _rotate_vec(SOCKET_ROT_XYZW, (0, 0, 1))
    print(f"\nSocket local X (insertion depth) -> world: ({socket_x_world[0]:.3f}, {socket_x_world[1]:.3f}, {socket_x_world[2]:.3f})")
    print(f"Socket local Y (height)          -> world: ({socket_y_world[0]:.3f}, {socket_y_world[1]:.3f}, {socket_y_world[2]:.3f})")
    print(f"Socket local Z (width)           -> world: ({socket_z_world[0]:.3f}, {socket_z_world[1]:.3f}, {socket_z_world[2]:.3f})")

    plug_x_world = _rotate_vec(PLUG_ROT_XYZW, (1, 0, 0))
    plug_y_world = _rotate_vec(PLUG_ROT_XYZW, (0, 1, 0))
    plug_z_world = _rotate_vec(PLUG_ROT_XYZW, (0, 0, 1))
    print(f"\nPlug local X -> world: ({plug_x_world[0]:.3f}, {plug_x_world[1]:.3f}, {plug_x_world[2]:.3f})")
    print(f"Plug local Y -> world: ({plug_y_world[0]:.3f}, {plug_y_world[1]:.3f}, {plug_y_world[2]:.3f})")
    print(f"Plug local Z -> world: ({plug_z_world[0]:.3f}, {plug_z_world[1]:.3f}, {plug_z_world[2]:.3f})")

    # --- 5. Compute insertion pose ---
    # Strategy: the plug needs to be oriented so its insertion axis aligns
    # with the socket insertion axis, and positioned so the connector part
    # is centered inside the socket cavity.
    #
    # Socket cavity center (local frame, from Body5 midpoint):
    #   Body5 X: [8.31, 13.16] -> midpoint ~10.7mm -> 0.0107m
    #   Body5 Y: [-1.04, 1.07] -> midpoint ~0.015mm -> ~0
    #   Body5 Z: [-3.30, 3.27] -> midpoint ~-0.015mm -> ~0
    # So the cavity center in socket local frame is approx (0.0107, 0, 0)
    #
    # Plug connector center (local frame, from Body1):
    #   Body1 X: [-2.85, 3.41] -> midpoint ~0.28mm -> 0.00028m
    #   Body1 Y: [-0.92, 0.92] -> midpoint ~0
    #   Body1 Z: [3.18, 12.98] -> midpoint ~8.08mm -> 0.00808m
    # The connector tip is at max Z in plug local frame: ~0.013m

    # Socket cavity entrance (high-X end in socket local frame)
    sock_cavity_entrance_local = (0.013, 0.0, 0.0)
    sock_cavity_center_local = (0.0107, 0.0, 0.0)

    # Transform to world
    cavity_entrance_world = _rotate_vec(SOCKET_ROT_XYZW, sock_cavity_entrance_local)
    cavity_entrance_world = tuple(c + s for c, s in zip(cavity_entrance_world, SOCKET_POS))

    cavity_center_world = _rotate_vec(SOCKET_ROT_XYZW, sock_cavity_center_local)
    cavity_center_world = tuple(c + s for c, s in zip(cavity_center_world, SOCKET_POS))

    print(f"\nSocket cavity entrance (world, mm): ({cavity_entrance_world[0]*1e3:.2f}, {cavity_entrance_world[1]*1e3:.2f}, {cavity_entrance_world[2]*1e3:.2f})")
    print(f"Socket cavity center  (world, mm): ({cavity_center_world[0]*1e3:.2f}, {cavity_center_world[1]*1e3:.2f}, {cavity_center_world[2]*1e3:.2f})")

    # The plug's connector tip (Body1 max-Z in plug local frame) should
    # align with the socket cavity entrance. We need the plug rotation such
    # that the plug's Z+ axis (connector insertion direction) aligns with
    # the socket's X- axis (insertion goes into the socket from the high-X side).
    #
    # Try several candidate rotations and pick the one that aligns axes best.
    # The plug connector extends along +Z in local frame.
    # The socket receptacle opens along -X in local frame (plug goes in from +X side).
    # So we need plug_local_Z to map to socket_local_(-X) in world frame.
    # i.e. plug_rot should map (0,0,1) to -socket_x_world

    target_insertion_dir = tuple(-c for c in socket_x_world)
    print(f"\nTarget insertion direction (world): ({target_insertion_dir[0]:.3f}, {target_insertion_dir[1]:.3f}, {target_insertion_dir[2]:.3f})")
    print(f"Current plug Z in world:           ({plug_z_world[0]:.3f}, {plug_z_world[1]:.3f}, {plug_z_world[2]:.3f})")

    # Try a grid of euler angles to find plug rotation that best aligns
    # plug_Z -> target_insertion_dir AND plug_Y -> socket_Y
    best_rot = None
    best_score = float('inf')
    for rx in range(0, 360, 90):
        for ry in range(0, 360, 90):
            for rz in range(0, 360, 90):
                rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
                # Build quaternion from euler XYZ
                cx, sx = math.cos(rx_r/2), math.sin(rx_r/2)
                cy, sy = math.cos(ry_r/2), math.sin(ry_r/2)
                cz, sz = math.cos(rz_r/2), math.sin(rz_r/2)
                # qx
                qx = (sx, 0, 0, cx)
                qy = (0, sy, 0, cy)
                qz = (0, 0, sz, cz)
                q = _quat_mul(_quat_mul(qz, qy), qx)  # ZYX convention

                pz = _rotate_vec(q, (0, 0, 1))
                py = _rotate_vec(q, (0, 1, 0))

                # Score: how well plug_Z aligns with target insertion dir
                dot_z = sum(a*b for a, b in zip(pz, target_insertion_dir))
                # Also check plug_Y aligns with socket_Y (height direction)
                dot_y = sum(a*b for a, b in zip(py, socket_y_world))
                score = (1 - dot_z)**2 + (1 - dot_y)**2
                if score < best_score:
                    best_score = score
                    best_rot = q
                    best_euler = (rx, ry, rz)

    print(f"\nBest plug rotation: euler=({best_euler[0]}, {best_euler[1]}, {best_euler[2]}) deg")
    print(f"  quaternion (x,y,z,w): ({best_rot[0]:.5f}, {best_rot[1]:.5f}, {best_rot[2]:.5f}, {best_rot[3]:.5f})")
    best_pz = _rotate_vec(best_rot, (0, 0, 1))
    best_py = _rotate_vec(best_rot, (0, 1, 0))
    best_px = _rotate_vec(best_rot, (1, 0, 0))
    print(f"  plug Z (insertion) -> world: ({best_pz[0]:.3f}, {best_pz[1]:.3f}, {best_pz[2]:.3f})")
    print(f"  plug Y (height)    -> world: ({best_py[0]:.3f}, {best_py[1]:.3f}, {best_py[2]:.3f})")
    print(f"  plug X             -> world: ({best_px[0]:.3f}, {best_px[1]:.3f}, {best_px[2]:.3f})")

    # Compute position: place plug connector tip at socket cavity entrance
    # Plug connector tip in plug local frame: center of Body1 at max Z
    plug_connector_tip_local = (0.00028, 0.0, 0.01298)
    tip_world = _rotate_vec(best_rot, plug_connector_tip_local)

    # Place tip at cavity entrance
    plug_goal_pos = tuple(ce - tw for ce, tw in zip(cavity_entrance_world, tip_world))

    print(f"\n--- COMPUTED INSERTION POSE ---")
    print(f"  plug_pos = ({plug_goal_pos[0]:.6f}, {plug_goal_pos[1]:.6f}, {plug_goal_pos[2]:.6f})")
    print(f"  plug_rot = ({best_rot[0]:.5f}, {best_rot[1]:.5f}, {best_rot[2]:.5f}, {best_rot[3]:.5f})")

    # Also compute a "partially inserted" pose (halfway into cavity)
    plug_partial_pos = tuple(ce - tw + 0.5*(cc - ce) for ce, tw, cc in
                             zip(cavity_entrance_world, tip_world, cavity_center_world))
    print(f"\n--- PARTIALLY INSERTED POSE ---")
    print(f"  plug_pos = ({plug_partial_pos[0]:.6f}, {plug_partial_pos[1]:.6f}, {plug_partial_pos[2]:.6f})")
    print(f"  plug_rot = ({best_rot[0]:.5f}, {best_rot[1]:.5f}, {best_rot[2]:.5f}, {best_rot[3]:.5f})")

    # --- 6. Cross-section fit check ---
    print("\n--- Cross-section fit analysis ---")
    # Transform plug points using the best rotation to socket local frame
    # so we can compare Y-Z cross sections directly
    sock_rot_inv = _quat_inv(SOCKET_ROT_XYZW)

    # Get socket points in socket-local frame (just the raw points)
    sock_local = socket_pts  # already in local frame

    # Transform plug to socket-local frame at the insertion pose
    plug_at_goal = _transform_points_by_pose(plug_pts, plug_goal_pos, best_rot)
    # Now convert from world to socket-local
    plug_in_sock_local = []
    for p in plug_at_goal:
        shifted = Gf.Vec3d(p[0]-SOCKET_POS[0], p[1]-SOCKET_POS[1], p[2]-SOCKET_POS[2])
        rp = _quat_to_matrix(sock_rot_inv).Transform(shifted)
        plug_in_sock_local.append(rp)

    # Check cross-sections at various X positions along socket insertion axis
    print(f"\n  Socket cavity (local Y-Z cross-section) vs Plug at insertion pose:")
    print(f"  {'X(mm)':>7s}  {'Sock Y range':>16s}  {'Sock Z range':>16s}  {'Plug Y range':>16s}  {'Plug Z range':>16s}  {'Fits?':>6s}")

    for x_mm in range(8, 16):
        x_m = x_mm / 1000.0
        tol = 0.001

        sock_cs = [(p[1], p[2]) for p in sock_local if abs(p[0] - x_m) < tol]
        plug_cs = [(p[1], p[2]) for p in plug_in_sock_local if abs(p[0] - x_m) < tol]

        if not sock_cs or not plug_cs:
            s_str = "no data" if not sock_cs else ""
            p_str = "no data" if not plug_cs else ""
            print(f"  {x_mm:7d}  {s_str:>16s}  {'':>16s}  {p_str:>16s}")
            continue

        sy = (min(p[0] for p in sock_cs)*1e3, max(p[0] for p in sock_cs)*1e3)
        sz = (min(p[1] for p in sock_cs)*1e3, max(p[1] for p in sock_cs)*1e3)
        py = (min(p[0] for p in plug_cs)*1e3, max(p[0] for p in plug_cs)*1e3)
        pz = (min(p[1] for p in plug_cs)*1e3, max(p[1] for p in plug_cs)*1e3)

        fits_y = py[0] >= sy[0] and py[1] <= sy[1]
        fits_z = pz[0] >= sz[0] and pz[1] <= sz[1]
        fits = "YES" if fits_y and fits_z else "NO"

        print(f"  {x_mm:7d}  [{sy[0]:6.2f},{sy[1]:6.2f}]  [{sz[0]:6.2f},{sz[1]:6.2f}]  "
              f"[{py[0]:6.2f},{py[1]:6.2f}]  [{pz[0]:6.2f},{pz[1]:6.2f}]  {fits:>6s}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"contact_offset was 0.02m (20mm) -- larger than the whole connector.")
    print(f"Reduced to 0.001m (1mm) in env config.")
    print(f"\nCopy these into displayport_basic_env_cfg.py or use 'i' command in interactive script:")
    print(f"  INSERTION_POSE_POS = ({plug_goal_pos[0]:.6f}, {plug_goal_pos[1]:.6f}, {plug_goal_pos[2]:.6f})")
    print(f"  INSERTION_POSE_ROT = ({best_rot[0]:.5f}, {best_rot[1]:.5f}, {best_rot[2]:.5f}, {best_rot[3]:.5f})")


if __name__ == "__main__":
    main()
