"""Headless diagnostic: dump cooked collision approximation for the DisplayPort
plug/socket and step the passive drop to measure insertion depth.

Usage:
    ./isaaclab.sh -p scripts/check_dp_collision.py --headless --steps 600
"""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

TASK = "Isaac-Deploy-DisplayportBasic-v0"
OUT = open("/home/shauryad/output_dir/dp_diag.txt", "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    OUT.write(msg + "\n")
    OUT.flush()

parser = argparse.ArgumentParser(description="Headless DisplayPort collision diagnostic.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument(
    "--insertion-pose",
    action="store_true",
    help="Teleport plug to the computed insertion pose (gravity off) and check it stays seated (tests the 'jumps out' instability).",
)
parser.add_argument(
    "--drop-aligned",
    action="store_true",
    help="Start the plug aligned and 15 mm above the seated pose, gravity ON, and check it slides in and seats (true dynamic insertion).",
)
# Computed by diagnose_dp_insertion.py for the current assets.
# diagnose prints the quat as (x,y,z,w)=(-0.70711,-0.70711,0,0); converted here
# to the (w,x,y,z) order that write_root_pose_to_sim expects.
INSERTION_POSE_POS = (0.00005, -0.01018, 0.16143)
INSERTION_POSE_ROT = (-0.5, 0.5, 0.5, 0.5)  # (w, x, y, z) — from find_dp_insertion.py
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def dump_collisions(stage):
    from pxr import Usd, UsdGeom, UsdPhysics

    for label, path in [
        ("PLUG", "/World/envs/env_0/DisplayPortPlug"),
        ("SOCKET", "/World/envs/env_0/DisplayPortSocket"),
    ]:
        root = stage.GetPrimAtPath(path)
        log(f"\n=== {label}: {path}  (valid={root.IsValid()}) ===")
        if not root.IsValid():
            continue
        pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for prim in Usd.PrimRange(root, pred):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                ke = prim.GetAttribute("physics:kinematicEnabled")
                log(f"  RigidBody root: {prim.GetPath()}  kinematic={ke.Get() if ke.IsValid() else None}")
        log(f"  root instanceable={root.IsInstanceable()}  is_instance={root.IsInstance()}")
        n_mesh = n_coll = 0
        for prim in Usd.PrimRange(root, pred):
            if prim.IsA(UsdGeom.Mesh):
                n_mesh += 1
            if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
                continue
            n_coll += 1
            approx = prim.GetAttribute("physics:approximation")
            ce = prim.GetAttribute("physics:collisionEnabled")
            physx = [s for s in prim.GetAppliedSchemas() if "Physx" in s and "Collision" in s]
            log(
                f"    {prim.GetParent().GetName()}/{prim.GetName()}"
                f"  approx={approx.Get() if approx.IsValid() else 'NA'}"
                f"  collEnabled={ce.Get() if ce.IsValid() else True}"
                f"  instProxy={prim.IsInstanceProxy()}"
                f"  physx={physx}"
            )
        log(f"  -> meshes(incl. instance proxies)={n_mesh}  collision_meshes={n_coll}")


def main():
    env_cfg, _ = resolve_task_config(TASK, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(TASK, cfg=env_cfg)
        env.reset()

        from isaaclab.sim.utils import get_current_stage

        dump_collisions(get_current_stage())

        plug = env.unwrapped.scene["dp_plug"]
        socket = env.unwrapped.scene["dp_socket"]
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        def _np(arr):
            try:
                return arr.numpy()
            except Exception:
                import warp as wp

                return wp.to_torch(arr).cpu().numpy()

        def report(tag):
            p = _np(plug.data.root_pos_w)[0].tolist()
            s = _np(socket.data.root_pos_w)[0].tolist()
            log(
                f"  [{tag:>10}] plug_z={p[2]:.4f}  socket_z={s[2]:.4f}  "
                f"dz(plug-socket)={p[2] - s[2]:+.4f}  plug_xy=({p[0]:+.4f},{p[1]:+.4f})"
            )

        if args_cli.insertion_pose:
            from pxr import PhysxSchema

            from isaaclab.sim.utils import get_current_stage

            # gravity off so we isolate collision response (ejection) from falling
            plug_prim = get_current_stage().GetPrimAtPath("/World/envs/env_0/DisplayPortPlug")
            PhysxSchema.PhysxRigidBodyAPI(plug_prim).GetDisableGravityAttr().Set(True)

            cur = plug.data.root_pos_w
            pose = _np(cur).copy()[0:1]  # placeholder shape
            pose_t = torch.tensor(
                [list(INSERTION_POSE_POS) + list(INSERTION_POSE_ROT)],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            plug.write_root_pose_to_sim(pose_t)
            zeros = torch.zeros((1, 6), device=env.unwrapped.device, dtype=torch.float32)
            plug.write_root_velocity_to_sim(zeros)

            log("\n=== insertion-pose stability (gravity OFF) — drift = ejection ===")
            start = _np(plug.data.root_pos_w)[0].copy()
            report("placed")
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    env.step(actions)
                    if (i + 1) % 60 == 0:
                        now = _np(plug.data.root_pos_w)[0]
                        drift = float(((now - start) ** 2).sum() ** 0.5)
                        report(f"step {i + 1}")
                        log(f"               drift_from_placed={drift * 1000:.2f} mm")
        elif args_cli.drop_aligned:
            # Start aligned to the insertion orientation, 15 mm above the seated
            # pose along the insertion axis (world +Z), gravity ON. Insertion
            # axis is world -Z, so gravity should drive the plug into the cavity.
            start_pos = (INSERTION_POSE_POS[0], INSERTION_POSE_POS[1], INSERTION_POSE_POS[2] + 0.015)
            pose_t = torch.tensor(
                [list(start_pos) + list(INSERTION_POSE_ROT)],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            plug.write_root_pose_to_sim(pose_t)
            plug.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=env.unwrapped.device, dtype=torch.float32)
            )
            seated_z = INSERTION_POSE_POS[2]
            log("\n=== aligned drop (gravity ON) — should slide DOWN to seated z≈%.4f ===" % seated_z)
            report("placed")
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    env.step(actions)
                    if (i + 1) % 60 == 0:
                        report(f"step {i + 1}")
            zf = _np(plug.data.root_pos_w)[0][2]
            log(f"  -> final plug_z={zf:.4f}  (seated≈{seated_z:.4f}; lower=inserted, ~0.20=rested on top)")
        else:
            log("\n=== drop test (plug falls under gravity) ===")
            report("t=0")
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    env.step(actions)
                    if (i + 1) % 60 == 0:
                        report(f"step {i + 1}")
            report("final")
    log("\nDONE")


if __name__ == "__main__":
    main()
