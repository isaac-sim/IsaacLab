# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp a real strawberry scan as a Newton VBD volume deformable and record how it deforms.

Swaps the task's procedural cuboid for the YCB ``012_strawberry`` scan (voxel-remeshed
to a watertight 1.2k-triangle surface by ``assets/make_strawberry_usd.py``), then runs
the same protocol as the other grasp scripts: the end-effector holds one constant pose
and only the gripper is actuated, at a constant commanded force.

Nodal positions are snapshotted through the squeeze so the deformation can be plotted
afterwards -- the bounding extents along each axis show the squeeze in y and the
Poisson bulge in x/z.

.. code-block:: bash

    isaaclab.bat -p scripts/environments/state_machine/grasp_strawberry.py --grip_effort 10 --youngs 2e5
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grasp a deformable strawberry with the Franka on Newton.")
parser.add_argument("--grip_effort", type=float, default=10.0, help="Finger effort cap [N] = grasp force.")
parser.add_argument(
    "--grip_stiffness",
    type=float,
    default=400.0,
    help=(
        "Finger joint stiffness. Keep it low enough that the drive is NOT saturated at the holding gap: a"
        " saturated actuator emits a constant force with its damping term clipped off, so the finger tips ring."
        " Use a high value (2e4) only when constant-force control matters more than a settled grasp."
    ),
)
parser.add_argument("--grip_damping", type=float, default=0.0, help="Finger joint damping (0 keeps the env's 100).")
parser.add_argument(
    "--grip_armature",
    type=float,
    default=0.5,
    help=(
        "Armature added to the finger joint inertia. This is the lever that actually stops the tips ringing:"
        " a bare Panda finger is ~15 g, so a saturated 10 N drive accelerates it at ~670 m/s^2 and it crosses"
        " its whole travel within one 1/60 s step, slamming and rebounding. Raising damping instead does not"
        " work -- damping is integrated explicitly, so it needs d < 2m/dt (~1.8 here) or the joint diverges."
    ),
)
parser.add_argument("--arm_stiffness", type=float, default=800.0, help="Arm joint stiffness (default ~80 misses).")
parser.add_argument("--arm_damping", type=float, default=80.0, help="Arm joint damping.")
parser.add_argument("--youngs", type=float, default=2e5, help="Young's modulus of the strawberry flesh [Pa].")
parser.add_argument("--poissons", type=float, default=0.3, help="Poisson's ratio [-].")
parser.add_argument("--density", type=float, default=900.0, help="Density [kg/m^3].")
parser.add_argument("--particle_radius", type=float, default=0.005, help="VBD particle radius [m].")
parser.add_argument(
    "--close_gap",
    type=float,
    default=0.0,
    help=(
        "If > 0, close the fingers to this fixed GAP [m] instead of squeezing to zero at a saturated force."
        " A force-saturated drive has no stable equilibrium against a springy fruit -- it compresses, is thrown"
        " back open, and limit-cycles (45 mm peak-to-peak). Commanding a position just inside the fruit gives a"
        " stable spring equilibrium instead. Set below the fruit width (~0.044 m) to compress it."
    ),
)
parser.add_argument(
    "--grip_ramp",
    type=int,
    default=150,
    help=(
        "Steps over which the grip force is ramped up to --grip_effort. A saturated actuator outputs a CONSTANT"
        " force: the d*(0 - qdot) damping term is clipped away, so nothing removes energy and the fingers slam"
        " into the fruit and bounce. Ramping keeps the drive gentle while contact is established. 0 = step input."
    ),
)
parser.add_argument(
    "--soft_contact_kd",
    type=float,
    default=1e-5,
    help=(
        "Soft-body contact damping (preset 1e-5 = essentially undamped). Raising it is tempting but DIVERGES:"
        " 1.0 blows the solver up within a few steps of contact and the arm Jacobian goes singular. Contact"
        " damping is integrated explicitly, so it is bounded by c < 2m/dt. Change only in small increments."
    ),
)
parser.add_argument("--shape_material_kd", type=float, default=1e-5, help="Rigid-shape contact damping (preset 1e-5).")
parser.add_argument("--substeps", type=int, default=10, help="Solver substeps per step (preset 10).")
parser.add_argument(
    "--physics_hz",
    type=float,
    default=240.0,
    help=(
        "Physics/actuator rate [Hz]. The preset runs 60 Hz with decimation 1, so the finger PD is recomputed"
        " only once per 1/60 s and a saturated 10 N drive pushes unopposed for that whole step -- far enough to"
        " overshoot the fruit and rebound. Solver substeps subdivide the integrator but NOT the actuator, so"
        " raising this is what actually calms the tips. The env step rate is held at 60 Hz via decimation."
    ),
)
parser.add_argument("--vbd_iterations", type=int, default=20, help="VBD solver iterations (preset 10).")
parser.add_argument("--edge_fac", type=float, default=0.12, help="Tet edge length factor.")
parser.add_argument("--pregrasp_steps", type=int, default=130, help="Steps above the fruit before descending.")
parser.add_argument("--pregrasp_z", type=float, default=0.1, help="Pre-grasp height above the COM [m].")
parser.add_argument("--settle_steps", type=int, default=140, help="Steps descending with the gripper open.")
parser.add_argument("--squeeze_steps", type=int, default=400, help="Steps with the gripper closing.")
parser.add_argument("--out", type=str, default=r"C:\isaac_soft\sweeps\strawberry.npz", help="Snapshot output.")
parser.add_argument("--video", action="store_true", default=False, help="Render the grasp through Newton's viewer.")
parser.add_argument("--video_folder", type=str, default="", help="Directory for the recording.")
parser.add_argument(
    "--cam_eye",
    type=str,
    default="0.66,-0.11,0.143",
    help="Camera position in ABSOLUTE world coords [m]. Headless runs have no viewport UI, so ViewerCfg's"
    " origin_type/asset tracking never applies and eye/lookat are taken as world positions -- the default"
    " (7.5,7.5,7.5)->(0,0,0) frames the robot base, leaving a 45 mm fruit invisible.",
)
parser.add_argument("--cam_lookat", type=str, default="0.5,0.0,0.073", help="Camera target, world coords [m].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# RecordVideo needs an rgb_array render mode, which in turn requires cameras to be enabled.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

import hashlib
import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg

from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Isaac-Lift-Soft-Franka-v0"
STRAWBERRY_USD = r"C:\isaac_soft\assets\strawberry_deformable.usda"
CACHE_DIR = r"C:\isaac_soft\.tetcache"
GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0


def _install_tet_cache(edge_fac: float):
    """Cache fTetWild's output so the strawberry tet mesh is identical on every run."""
    import pytetwild

    original = pytetwild.tetrahedralize
    os.makedirs(CACHE_DIR, exist_ok=True)

    def cached(vertices, faces, **kwargs):
        kwargs["edge_length_fac"] = edge_fac
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
        h.update(repr(sorted(kwargs.items())).encode())
        path = os.path.join(CACHE_DIR, f"tet_{h.hexdigest()[:16]}.npz")
        if os.path.exists(path):
            d = np.load(path)
            return d["points"], d["indices"]
        points, indices = original(vertices, faces, **kwargs)
        np.savez(path, points=points, indices=indices)
        return points, indices

    pytetwild.tetrahedralize = cached


_install_tet_cache(args_cli.edge_fac)


def lame(youngs, poissons):
    return (
        youngs / (2.0 * (1.0 + poissons)),
        youngs * poissons / ((1.0 + poissons) * (1.0 - 2.0 * poissons)),
    )


def extents(nodal):
    """Axis-aligned bounding extents [m] of one env's nodal cloud."""
    return (nodal.max(axis=0) - nodal.min(axis=0)).tolist()


def main():
    if not os.path.exists(STRAWBERRY_USD):
        raise FileNotFoundError(f"strawberry asset missing: {STRAWBERRY_USD} (run assets/make_strawberry_usd.py)")

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=[])  # Newton MJWarp + VBD
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = 1
    cfg.seed = 0

    # raise the actuator rate while holding the env step at 60 Hz
    if args_cli.physics_hz > 0:
        env_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
        cfg.sim.dt = 1.0 / args_cli.physics_hz
        cfg.decimation = max(1, int(round(args_cli.physics_hz / env_hz)))
        cfg.sim.render_interval = cfg.decimation
        print(f"[BERRY] physics {args_cli.physics_hz:g} Hz, decimation {cfg.decimation}, env {env_hz:g} Hz", flush=True)

    trial = args_cli.pregrasp_steps + args_cli.settle_steps + args_cli.squeeze_steps
    cfg.episode_length_s = max(40.0, (trial + 60) * cfg.sim.dt * cfg.decimation)

    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_deformable.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

    # swap the procedural cuboid for the scanned strawberry
    k_mu, k_lambda = lame(args_cli.youngs, args_cli.poissons)
    cfg.scene.deformable = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=STRAWBERRY_USD,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=args_cli.density,
                k_mu=k_mu,
                k_lambda=k_lambda,
                particle_radius=args_cli.particle_radius,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.06, 0.13), roughness=0.45),
        ),
    )

    # frame the fruit. Absolute world coords: headless has no viewport UI, so asset tracking
    # never kicks in and these are used verbatim.
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
    print(f"[BERRY] camera eye={cfg.viewer.eye} lookat={cfg.viewer.lookat}", flush=True)

    cfg.scene.robot.actuators["panda_hand"].effort_limit_sim = args_cli.grip_effort
    if args_cli.close_gap > 0.0:
        # position-hold grasp: target each finger at half the desired gap, and leave enough
        # effort headroom that the drive never saturates (saturation is what kills the damping)
        cfg.actions.gripper_action.close_command_expr = {"panda_finger_.*": args_cli.close_gap / 2.0}
        cfg.scene.robot.actuators["panda_hand"].effort_limit_sim = max(args_cli.grip_effort, 100.0)
        print(f"[BERRY] position-hold grasp: target gap {args_cli.close_gap:.4f} m", flush=True)
    cfg.scene.robot.actuators["panda_hand"].stiffness = args_cli.grip_stiffness
    if args_cli.grip_damping > 0.0:
        cfg.scene.robot.actuators["panda_hand"].damping = args_cli.grip_damping
    if args_cli.grip_armature > 0.0:
        cfg.scene.robot.actuators["panda_hand"].armature = args_cli.grip_armature

    # damp the contact and tighten the solver: the preset's 1e-5 damping makes contact
    # essentially elastic, which is what the finger tips ring against
    model = getattr(cfg.sim.physics, "model_cfg", None)
    if model is not None:
        model.soft_contact_kd = args_cli.soft_contact_kd
        model.shape_material_kd = args_cli.shape_material_kd
    if args_cli.substeps > 0 and hasattr(cfg.sim.physics, "num_substeps"):
        cfg.sim.physics.num_substeps = args_cli.substeps
    solver = getattr(cfg.sim.physics, "solver_cfg", None)
    soft_solver = getattr(solver, "soft_solver_cfg", None) if solver is not None else None
    if soft_solver is not None and args_cli.vbd_iterations > 0:
        soft_solver.iterations = args_cli.vbd_iterations
    print(
        f"[BERRY] contact damping soft_kd={args_cli.soft_contact_kd} shape_kd={args_cli.shape_material_kd} "
        f"substeps={args_cli.substeps} vbd_iters={args_cli.vbd_iterations} ramp={args_cli.grip_ramp}",
        flush=True,
    )
    for name, act in cfg.scene.robot.actuators.items():
        if name != "panda_hand":
            act.stiffness = args_cli.arm_stiffness
            act.damping = args_cli.arm_damping

    print(f"[BERRY] asset={STRAWBERRY_USD} E={args_cli.youngs:g} nu={args_cli.poissons} F={args_cli.grip_effort} N")

    # Newton renders through its own pyglet/OpenGL viewer, whose headless path uses EGL.
    # EGL does not exist on Windows ("Library 'EGL' not found"), so force a windowed WGL
    # context to make frame readback work -- a viewer window will pop up while recording.
    if args_cli.video:
        import newton.viewer as _nv

        _OrigViewerGL = _nv.ViewerGL

        def _WindowedViewerGL(*a, **k):
            import pyglet

            pyglet.options["headless"] = False
            k["headless"] = False
            return _OrigViewerGL(*a, **k)

        _nv.ViewerGL = _WindowedViewerGL
        print("[BERRY] patched Newton ViewerGL -> windowed (WGL) for capture on Windows", flush=True)

    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        folder = args_cli.video_folder or r"C:\isaac_soft\videos\strawberry"
        os.makedirs(folder, exist_ok=True)
        total = args_cli.pregrasp_steps + args_cli.settle_steps + args_cli.squeeze_steps
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=folder,
            step_trigger=lambda step: step == 0,
            video_length=total,
            name_prefix="grasp_strawberry",
            disable_logger=True,
        )
        print(f"[BERRY] recording to {folder}", flush=True)

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger.*")
    berry = env.unwrapped.scene["deformable"]

    env.reset(seed=0)
    nodal0 = berry.data.nodal_pos_w.torch[0].clone()
    print(f"[BERRY] tet nodes={nodal0.shape[0]} rest_extents={np.round(extents(nodal0.cpu().numpy()), 5).tolist()}")

    com = berry.data.root_pos_w.torch.clone()
    com_local = com - env.unwrapped.scene.env_origins
    actions = torch.zeros((1, env.unwrapped.action_space.shape[-1]), device=device)
    actions[:, 0:3] = com_local
    actions[:, 3] = 1.0
    actions[:, -1] = GRIPPER_OPEN

    # approach from above, then descend
    actions[:, 2] = com_local[:, 2] + args_cli.pregrasp_z
    for _ in range(args_cli.pregrasp_steps):
        env.step(actions)
    actions[:, 2] = com_local[:, 2]
    for _ in range(args_cli.settle_steps):
        env.step(actions)

    rest = berry.data.nodal_pos_w.torch[0].clone().cpu().numpy()
    ee_w = env.unwrapped.scene["ee_frame"].data.target_pos_w.torch[0, 0, :]
    print(f"[BERRY] pre-squeeze extents={np.round(extents(rest), 5).tolist()} "
          f"miss={np.round((ee_w - berry.data.root_pos_w.torch[0]).abs().cpu().numpy(), 4).tolist()}")

    # squeeze at constant force
    actions[:, -1] = GRIPPER_CLOSE
    frames, times, gaps, efforts, ext_log = [], [], [], [], []
    ramp_from = 0.35  # start at a fraction of the target force, not zero, or the fingers never close
    for k in range(args_cli.squeeze_steps):
        if args_cli.grip_ramp > 0:
            frac = min(1.0, (k + 1) / args_cli.grip_ramp)
            limit = args_cli.grip_effort * (ramp_from + (1.0 - ramp_from) * frac)
            robot.write_joint_effort_limit_to_sim_index(limits=limit, joint_ids=finger_ids)
        env.step(actions)
        nodal = berry.data.nodal_pos_w.torch[0].cpu().numpy()
        gaps.append(float(robot.data.joint_pos[0, finger_ids].sum().item()))
        efforts.append(float(robot.data.applied_torque[0, finger_ids].abs().max().item()))
        ext_log.append(extents(nodal))
        if k % 40 == 0 or k == args_cli.squeeze_steps - 1:
            frames.append(nodal.copy())
            times.append(k)
            print(
                f"[BERRY] step={k:3d} gap={gaps[-1]:.5f} F={efforts[-1]:.2f} "
                f"extents={np.round(ext_log[-1], 5).tolist()}",
                flush=True,
            )

    ext_log = np.array(ext_log)
    rest_ext = np.array(extents(rest))
    final_ext = ext_log[-1]
    print(
        f"[BERRY] RESULT rest_width_y={rest_ext[1]:.5f} squeezed_width_y={final_ext[1]:.5f} "
        f"squeeze_y={(rest_ext[1] - final_ext[1]) * 1e3:+.2f}mm "
        f"bulge_x={(final_ext[0] - rest_ext[0]) * 1e3:+.2f}mm bulge_z={(final_ext[2] - rest_ext[2]) * 1e3:+.2f}mm "
        f"final_gap={gaps[-1]:.5f} applied={efforts[-1]:.2f}",
        flush=True,
    )

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez(
        args_cli.out,
        rest=rest,
        frames=np.array(frames),
        frame_steps=np.array(times),
        gaps=np.array(gaps),
        efforts=np.array(efforts),
        extents=ext_log,
        rest_extents=rest_ext,
        youngs=args_cli.youngs,
        force=args_cli.grip_effort,
    )
    print(f"[BERRY] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
