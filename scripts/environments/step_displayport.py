"""Interactive runner for DisplayPort basic drop test.

Usage:
    ./isaaclab.sh -p scripts/environments/step_displayport.py --num_envs 1 --viz kit

The simulation runs continuously at a slow rate. Commands are read
from stdin in a non-blocking way so the GUI stays fully interactive
(zoom, pan, orbit all work normally).

Commands (type at the prompt, then press Enter):
    x/y/z <val>    move plug by <val> metres along that axis  (e.g. "x 0.01")
    p              print current plug & socket positions
    g              toggle gravity on the plug (disable to position it freely)
    r              reset scene to initial state
    f              fast mode (normal sim speed)
    s              slow mode (default — pauses between steps)
    q              quit
"""

import argparse
import select
import sys
import time

import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

TASK = "Isaac-Deploy-DisplayportBasic-v0"

# Insertion pose: plug tip at socket opening (simready assets)
# Recompute with diagnose_dp_insertion.py if assets change
INSERTION_POSE_POS = (0.0, 0.066, 0.2305)
INSERTION_POSE_ROT = (0.70711, 0.70711, 0.00000, 0.00000)

parser = argparse.ArgumentParser(description="Interactive DisplayPort drop test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_delay", type=float, default=0.5,
                    help="Seconds between physics steps in slow mode (default 0.5).")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

HELP = """
=== Interactive plug positioning (GUI stays responsive) ===
  x <val>        → move plug along X  (e.g. "x 0.01"  or  "x -0.005")
  y <val>        → move plug along Y
  z <val>        → move plug along Z
  rx <deg>       → rotate plug about world X  (e.g. "rx 90")
  ry <deg>       → rotate plug about world Y
  rz <deg>       → rotate plug about world Z
  p              → print plug & socket positions
  c              → inspect collision mesh types on all prims
  t              → show active contact points (which bodies are colliding & where)
  d <body>       → disable collision on a socket body (e.g. "d Body8")
  e <body>       → enable collision on a socket body (e.g. "e Body8")
  h <body>       → toggle visibility of a socket body (e.g. "h Body4")
  i              → teleport plug to computed insertion pose (gravity off)
  g              → toggle gravity on plug (off = free positioning)
  k              → freeze/unfreeze plug (toggle kinematic — stops all movement)
  r              → reset scene
  f              → fast mode (full speed)
  s              → slow mode (paused between steps, default)
  q              → quit

Simulation runs continuously — use mouse in the Kit window to
zoom / pan / orbit freely.
"""


def _get_plug(env):
    return env.unwrapped.scene["dp_plug"]


def _get_socket(env):
    return env.unwrapped.scene["dp_socket"]


def _print_poses(env):
    plug = _get_plug(env)
    sock = _get_socket(env)
    plug_pos = wp.to_torch(plug.data.root_pos_w)[0].cpu()
    plug_rot = wp.to_torch(plug.data.root_quat_w)[0].cpu()
    sock_pos = wp.to_torch(sock.data.root_pos_w)[0].cpu()
    sock_rot = wp.to_torch(sock.data.root_quat_w)[0].cpu()
    print(f"  Plug   pos={plug_pos.tolist()}  rot={plug_rot.tolist()}")
    print(f"  Socket pos={sock_pos.tolist()}  rot={sock_rot.tolist()}")


def _move_plug(env, axis: int, delta: float):
    """Teleport plug by *delta* metres along the given axis (0=x, 1=y, 2=z)."""
    plug = _get_plug(env)
    pos = wp.to_torch(plug.data.root_pos_w).clone()
    quat = wp.to_torch(plug.data.root_quat_w).clone()
    pos[0, axis] += delta
    plug.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))


def _rotate_plug(env, axis: int, deg: float):
    """Rotate plug by *deg* degrees about the given WORLD axis (0=x, 1=y, 2=z)."""
    import math

    plug = _get_plug(env)
    pos = wp.to_torch(plug.data.root_pos_w).clone()
    q = wp.to_torch(plug.data.root_quat_w).clone()[0]  # (w, x, y, z)
    half = math.radians(deg) / 2.0
    d = [math.cos(half), 0.0, 0.0, 0.0]
    d[axis + 1] = math.sin(half)
    dq = torch.tensor(d, device=q.device, dtype=q.dtype)
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = q
    # world-frame rotation: new = dq (x) q
    new = torch.tensor(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        device=q.device,
        dtype=q.dtype,
    ).unsqueeze(0)
    plug.write_root_pose_to_sim(torch.cat([pos, new], dim=-1))


def _toggle_gravity(env):
    """Toggle gravity on the plug rigid body."""
    from pxr import PhysxSchema
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    plug_prim = stage.GetPrimAtPath("/World/envs/env_0/DisplayPortPlug")
    try:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(plug_prim)
        cur = physx_rb.GetDisableGravityAttr().Get()
        physx_rb.GetDisableGravityAttr().Set(not cur)
        state = "OFF" if not cur else "ON"
    except Exception:
        state = "toggle failed"
    print(f"  Plug gravity: {state}")


def _set_gravity(env, disable: bool):
    """Explicitly set gravity on the plug."""
    from pxr import PhysxSchema
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    plug_prim = stage.GetPrimAtPath("/World/envs/env_0/DisplayPortPlug")
    try:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(plug_prim)
        physx_rb.GetDisableGravityAttr().Set(disable)
    except Exception:
        pass


def _freeze_plug(env):
    """Toggle kinematic mode on the plug to freeze/unfreeze it in place."""
    from pxr import UsdPhysics
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    plug_prim = stage.GetPrimAtPath("/World/envs/env_0/DisplayPortPlug")
    rb = UsdPhysics.RigidBodyAPI(plug_prim)
    kinematic_attr = rb.GetKinematicEnabledAttr()
    cur = kinematic_attr.Get() if kinematic_attr.IsValid() else False
    kinematic_attr.Set(not cur)
    if not cur:
        plug = _get_plug(env)
        zeros = torch.zeros(1, 3, device=wp.to_torch(plug.data.root_pos_w).device)
        plug.write_root_velocity_to_sim(torch.cat([zeros, zeros], dim=-1))
    state = "FROZEN (kinematic)" if not cur else "UNFROZEN (dynamic)"
    print(f"  Plug: {state}")


def _insert_plug(env):
    """Teleport plug to the computed insertion pose with gravity disabled."""
    _set_gravity(env, disable=True)
    plug = _get_plug(env)
    cur_pos = wp.to_torch(plug.data.root_pos_w).clone()
    pose_vals = list(INSERTION_POSE_POS) + list(INSERTION_POSE_ROT)
    pose = cur_pos.new_tensor(pose_vals).unsqueeze(0)
    plug.write_root_pose_to_sim(pose)
    zeros = torch.zeros_like(cur_pos)
    plug.write_root_velocity_to_sim(torch.cat([zeros, zeros], dim=-1))
    print("  Plug teleported to insertion pose (gravity OFF).")
    _print_poses(env)


def _inspect_collisions(env):
    """Walk the prim tree for plug and socket and print collision mesh info."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    for label, path in [("PLUG", "/World/envs/env_0/DisplayPortPlug"),
                        ("SOCKET", "/World/envs/env_0/DisplayPortSocket")]:
        print(f"\n  === {label}: {path} ===")
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            print(f"    (prim not found)")
            continue
        for prim in Usd.PrimRange(root):
            ppath = str(prim.GetPath())
            ptype = prim.GetTypeName()
            is_mesh = prim.IsA(UsdGeom.Mesh)
            has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
            has_mesh_col = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            schemas_list = prim.GetAppliedSchemas()

            extras = []
            if has_rb:
                extras.append("RigidBody")
            if has_col:
                col_enabled = prim.GetAttribute("physics:collisionEnabled")
                enabled_str = "ON" if (not col_enabled.IsValid() or col_enabled.Get() != False) else "OFF"
                extras.append(f"Collision({enabled_str})")
            if has_mesh_col:
                approx_attr = prim.GetAttribute("physics:approximation")
                approx = approx_attr.Get() if approx_attr.IsValid() else "N/A"
                extras.append(f"MeshCollision(approx={approx})")

            physx_schemas = [s for s in schemas_list if "Physx" in s and "Collision" in s]
            if physx_schemas:
                extras.append(f"PhysX={physx_schemas}")

            tag = " [MESH]" if is_mesh else ""
            apis = ", ".join(extras) if extras else "none"
            if is_mesh and has_col:
                print(f"    {ppath}{tag}  apis=[{apis}]")
    print()


def _toggle_body_collision(env, body_name: str, enable: bool):
    """Enable or disable collision on a specific socket body by name."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    socket_path = "/World/envs/env_0/DisplayPortSocket"
    root = stage.GetPrimAtPath(socket_path)
    if not root.IsValid():
        print(f"  Socket prim not found.")
        return

    found = False
    for prim in Usd.PrimRange(root):
        parent_name = prim.GetParent().GetName() if prim.GetParent() else ""
        if parent_name == body_name and prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            col_attr = prim.GetAttribute("physics:collisionEnabled")
            if not col_attr.IsValid():
                col_attr = prim.CreateAttribute("physics:collisionEnabled", prim.GetStage().GetRootLayer().GetPropertyAtPath(prim.GetPath().AppendProperty("physics:collisionEnabled")).typeName if False else None)
            prim.GetAttribute("physics:collisionEnabled").Set(enable)
            state = "ENABLED" if enable else "DISABLED"
            print(f"  {body_name}: collision {state} on {prim.GetPath()}")
            found = True
    if not found:
        print(f"  Body '{body_name}' not found in socket. Available bodies:")
        bodies = set()
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                bodies.add(prim.GetParent().GetName())
        print(f"    {', '.join(sorted(bodies))}")


def _toggle_body_visibility(env, body_name: str):
    """Toggle visibility of a socket body to visually identify it."""
    from pxr import Usd, UsdGeom
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    socket_path = "/World/envs/env_0/DisplayPortSocket"
    root = stage.GetPrimAtPath(socket_path)
    if not root.IsValid():
        print("  Socket prim not found.")
        return

    found = False
    for prim in Usd.PrimRange(root):
        parent_name = prim.GetParent().GetName() if prim.GetParent() else ""
        if parent_name == body_name and prim.IsA(UsdGeom.Mesh):
            imageable = UsdGeom.Imageable(prim)
            cur = imageable.ComputeVisibility()
            if cur == UsdGeom.Tokens.invisible:
                imageable.MakeVisible()
                print(f"  {body_name}: now VISIBLE")
            else:
                imageable.MakeInvisible()
                print(f"  {body_name}: now HIDDEN")
            found = True
            break
    if not found:
        print(f"  Body '{body_name}' not found.")


def _show_contacts(env):
    """Show contact forces on the plug using Isaac Lab's built-in data."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from isaaclab.sim.utils import get_current_stage
    import omni.physx
    from carb import Float3

    plug = _get_plug(env)
    plug_pos = wp.to_torch(plug.data.root_pos_w)[0].cpu().tolist()

    # Try various ways to get contact force
    force_found = False
    try:
        # With activate_contact_sensors=True, this should work
        force_data = plug.data.net_forces_w
        if force_data is not None:
            net_force = wp.to_torch(force_data)[0].cpu().tolist()
            force_mag = (net_force[0]**2 + net_force[1]**2 + net_force[2]**2)**0.5
            print(f"  Plug net force: ({net_force[0]:.4f}, {net_force[1]:.4f}, {net_force[2]:.4f}) N  |F|={force_mag:.4f} N")
            force_found = True
            if force_mag > 1e-6:
                fx, fy, fz = net_force[0]/force_mag, net_force[1]/force_mag, net_force[2]/force_mag
                print(f"  → Force direction: ({fx:.3f}, {fy:.3f}, {fz:.3f})")
                if fz > 0.5:
                    print(f"  → Plug is being pushed UPWARD (blocked from below)")
                if abs(fx) > 0.3 or abs(fy) > 0.3:
                    print(f"  → Plug has lateral contact (misaligned)")
    except Exception:
        pass

    if not force_found:
        # List available data attributes for debugging
        try:
            data_attrs = [a for a in dir(plug.data) if not a.startswith('_')
                         and ('force' in a.lower() or 'contact' in a.lower())]
            print(f"  Available force/contact attrs: {data_attrs}")
        except Exception:
            pass

    print(f"\n  Plug pos: ({plug_pos[0]*1e3:.2f}, {plug_pos[1]*1e3:.2f}, {plug_pos[2]*1e3:.2f}) mm")

    # Overlap sphere query with correct API
    try:
        physx_sq = omni.physx.get_physx_scene_query_interface()

        hits = []
        def report_hit(hit):
            hits.append(str(hit.rigid_body))
            return True

        pos = Float3(plug_pos[0], plug_pos[1], plug_pos[2])
        physx_sq.overlap_sphere(0.02, pos, report_hit, False)

        if hits:
            print(f"\n  Bodies within 20mm of plug center:")
            for h in hits:
                print(f"    → {h}")
        else:
            print(f"\n  No bodies found within 20mm sphere of plug center.")
    except Exception as e:
        print(f"\n  Overlap query failed: {e}")

    print(f"\n  TIP: To identify the blocker, try:")
    print(f"    1. Freeze plug with 'k'")
    print(f"    2. Disable bodies one at a time: 'd Body4', 'd Body5', etc.")
    print(f"    3. Unfreeze with 'k' — if plug moves further, that body was blocking")


def _check_stdin():
    """Return a line from stdin if available, else None (non-blocking)."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()
    return None


def _handle_command(raw, env):
    """Process one command. Returns True if should quit."""
    cmd = raw.lower().split()
    if not cmd:
        return False

    axis_map = {"x": 0, "y": 1, "z": 2}

    if cmd[0] == "q":
        return True
    elif cmd[0] == "i":
        _insert_plug(env)
    elif cmd[0] == "p":
        _print_poses(env)
    elif cmd[0] == "c":
        _inspect_collisions(env)
    elif cmd[0] == "t":
        _show_contacts(env)
    elif cmd[0] == "d":
        if len(cmd) < 2:
            print("  Usage: d <BodyName>  (e.g. 'd Body8')")
        else:
            _toggle_body_collision(env, raw.split()[1], enable=False)
    elif cmd[0] == "e":
        if len(cmd) < 2:
            print("  Usage: e <BodyName>  (e.g. 'e Body8')")
        else:
            _toggle_body_collision(env, raw.split()[1], enable=True)
    elif cmd[0] == "h":
        if len(cmd) < 2:
            print("  Usage: h <BodyName>  (toggle visibility, e.g. 'h Body4')")
        else:
            _toggle_body_visibility(env, raw.split()[1])
    elif cmd[0] == "g":
        _toggle_gravity(env)
    elif cmd[0] == "k":
        _freeze_plug(env)
    elif cmd[0] == "r":
        env.reset()
        print("  Scene reset.")
        _print_poses(env)
    elif cmd[0] == "f":
        print("  Fast mode.")
        return "fast"
    elif cmd[0] == "s":
        print("  Slow mode.")
        return "slow"
    elif cmd[0] in ("rx", "ry", "rz"):
        if len(cmd) < 2:
            print(f"  Usage: {cmd[0]} <deg>  (rotate plug about world {cmd[0][1]}-axis, e.g. {cmd[0]} 90)")
        else:
            try:
                _rotate_plug(env, {"rx": 0, "ry": 1, "rz": 2}[cmd[0]], float(cmd[1]))
                _print_poses(env)
            except ValueError:
                print(f"  Invalid value: {cmd[1]}")
    elif cmd[0] in axis_map:
        if len(cmd) < 2:
            print(f"  Usage: {cmd[0]} <value>  (e.g. {cmd[0]} 0.01)")
        else:
            try:
                delta = float(cmd[1])
                _move_plug(env, axis_map[cmd[0]], delta)
                _print_poses(env)
            except ValueError:
                print(f"  Invalid value: {cmd[1]}")
    else:
        print("  Unknown command. Type 'q' to quit, 'p' for poses, 'x/y/z <val>' to move plug.")
    return False


def main():
    env_cfg, _ = resolve_task_config(TASK, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(TASK, cfg=env_cfg)
        env.reset()

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step_count = 0
        slow_mode = True
        step_delay = args_cli.step_delay

        print(HELP)
        _print_poses(env)
        print(f"\nSimulation running (slow mode, {step_delay}s/step). Type commands anytime.\n")

        while True:
            line = _check_stdin()
            if line is not None:
                result = _handle_command(line, env)
                if result is True:
                    break
                elif result == "fast":
                    slow_mode = False
                elif result == "slow":
                    slow_mode = True

            with torch.inference_mode():
                env.step(actions)
                step_count += 1

            if slow_mode:
                time.sleep(step_delay)

        env.close()


if __name__ == "__main__":
    main()
