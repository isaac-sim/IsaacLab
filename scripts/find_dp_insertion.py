"""Find the plug orientation+position that nests the plug blade in the socket
cavity, working in the LIVE env (so every transform - env, init_state, asset
internal, scale - is baked in).

Method: express the socket cavity (Body4 u Body5) AABB in socket-root-local
space, and the plug blade (Body1) points in plug-root-local space. For each of
the 24 axis-aligned rotations, rotate the blade, align its centroid to the
cavity centroid, and count the fraction of blade points that fall inside the
cavity AABB. The best rotation is the mating orientation. Report the plug WORLD
quaternion (w,x,y,z) and seated position.

Usage:
    ./isaaclab.sh -p scripts/find_dp_insertion.py --headless
"""

import argparse
import itertools
import sys

import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

TASK = "Isaac-Deploy-DisplayportBasic-v0"
OUT = open("/home/shauryad/output_dir/dp_find.txt", "w")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m)
    OUT.write(m + "\n")
    OUT.flush()


def rot_group_24():
    """Yield the 24 proper (det=+1) axis-aligned rotation matrices."""
    seen = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                seen.append(M)
    return seen


def mat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def body_world_points(stage, root_path, body_names):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    pts = []
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root, pred):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.GetParent().GetName() not in body_names:
            continue
        local = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        m = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64).reshape(4, 4)
        w = (np.c_[local, np.ones(len(local))] @ m)[:, :3]
        pts.append(w)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def root_world_matrix(stage, root_path):
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    return np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path)), dtype=np.float64).reshape(4, 4)


def main():
    import gymnasium as gym

    env_cfg, _ = resolve_task_config(TASK, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(TASK, cfg=env_cfg)
        env.reset()
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        plug_path = "/World/envs/env_0/DisplayPortPlug"
        sock_path = "/World/envs/env_0/DisplayPortSocket"

        # socket-root world frame
        Ts = root_world_matrix(stage, sock_path)
        Rs = Ts[:3, :3]
        Rs = Rs / np.linalg.norm(Rs, axis=0)  # normalize (strip scale) columns
        ts = Ts[3, :3] if Ts.shape == (4, 4) else Ts[:3, 3]
        # USD row-vector convention: world = [p,1] @ M, translation in M[3,:3]
        ts = Ts[3, :3]

        def to_socket_local(world_pts):
            return (world_pts - ts) @ Rs  # Rs columns are socket axes in world

        cavity_w = body_world_points(stage, sock_path, {"Body4", "Body5"})
        blade_w = body_world_points(stage, plug_path, {"Body1"})
        log(f"cavity points={len(cavity_w)}  blade points={len(blade_w)}")

        cav_local = to_socket_local(cavity_w)
        cav_min, cav_max = cav_local.min(0), cav_local.max(0)
        cav_ctr = 0.5 * (cav_min + cav_max)
        cav_ext = cav_max - cav_min
        log(f"cavity socket-local AABB (mm): min={np.round(cav_min*1000,2)} max={np.round(cav_max*1000,2)} ext={np.round(cav_ext*1000,2)}")

        # blade in plug-root-local
        Tp = root_world_matrix(stage, plug_path)
        Rp = Tp[:3, :3]
        Rp = Rp / np.linalg.norm(Rp, axis=0)
        tp = Tp[3, :3]
        blade_plocal = (blade_w - tp) @ Rp
        blade_pc = blade_plocal.mean(0)

        best = None
        for R in rot_group_24():
            # blade expressed in socket-local if plug-root is rotated by R relative to socket
            b = (blade_plocal - blade_pc) @ R.T  # rotate
            b = b - b.mean(0) + cav_ctr  # align centroid to cavity center
            inside = np.all((b >= cav_min - 0.0005) & (b <= cav_max + 0.0005), axis=1)
            frac = inside.mean()
            ext = b.max(0) - b.min(0)
            # prefer high containment AND blade longer than cavity along one axis (it protrudes out the opening)
            if best is None or frac > best[1]:
                best = (R, frac, ext)

        R_best, frac, ext = best
        log(f"\nbest blade-in-cavity fraction={frac:.3f}  blade ext in socket-local(mm)={np.round(ext*1000,2)}")
        # plug world rotation = socket world rotation * R_best
        Rworld = Rs @ R_best
        q = mat_to_quat_wxyz(Rworld)
        log(f"plug WORLD quat (w,x,y,z) = ({q[0]:+.5f}, {q[1]:+.5f}, {q[2]:+.5f}, {q[3]:+.5f})")

        # seated position: place plug root so blade centroid -> cavity centroid (world),
        # backed off slightly along world up (+Z) so a gravity drop finishes the insert.
        blade_ctr_world_offset = (Rworld @ (blade_pc)) if False else None
        cav_ctr_world = cav_ctr @ Rs.T + ts
        # plug root must be placed so that (R_best applied) blade centroid lands at cavity center.
        # world position of plug root = cav_ctr_world - Rworld @ blade_pc
        plug_pos = cav_ctr_world - Rworld @ blade_pc
        log(f"seated plug WORLD pos (m) = ({plug_pos[0]:+.5f}, {plug_pos[1]:+.5f}, {plug_pos[2]:+.5f})")
        log(f"  (socket origin world z={ts[2]:.4f}; cavity center world z={cav_ctr_world[2]:.4f})")
    log("\nDONE")


if __name__ == "__main__":
    main()
