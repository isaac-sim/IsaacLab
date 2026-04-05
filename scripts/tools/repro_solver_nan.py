# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone MuJoCo Warp solver NaN reproduction.

Replays a NaN debug export (ring npz + companion USD) to reproduce the failure.

The ring npz contains ALL envs' data across 200 frames:
  - joint_q, joint_qd, body_q, body_qd (full flat arrays per frame)
  - diag_qfrc_applied (per-world per-step forces)
  - cfg_* keys (solver + collision pipeline config)
  - exported_env_ids (which env(s) NaN'd)

Replay strategy: for each frame in the ring, reset the simulation to the
recorded joint state, apply the recorded forces, step once, and check if
that step produces NaN.  This tests each step independently with the exact
inputs from training.

Usage:
    python scripts/tools/repro_solver_nan.py nan_debug/nan_replay_<ts>.npz
    python scripts/tools/repro_solver_nan.py nan_debug/nan_replay_<ts>.npz --nan_only --visualize
"""

from __future__ import annotations

import argparse
import re
import sys
import time as _time

import numpy as np
import torch
import warp as wp

from newton import CollisionPipeline, ModelBuilder, eval_fk
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverMuJoCo


def _find_valid_range(body_q: np.ndarray, bad_world: int = -1, bodies_per_env: int = 17) -> tuple[int, int]:
    """Return (first_valid, last_valid) frame indices.

    For full-world exports, checks only the bad_world's bodies.
    """
    n = body_q.shape[0]
    if bad_world >= 0 and body_q.ndim == 3:
        b0 = bad_world * bodies_per_env
        b1 = b0 + bodies_per_env
        check = body_q[:, b0:b1, :]
    else:
        check = body_q
    first = next((s for s in range(n) if np.isfinite(check[s]).all()), 0)
    last = next((s for s in range(n - 1, -1, -1) if np.isfinite(check[s]).all()), 0)
    return first, last


def _build_model(usd_path: str, world_positions: np.ndarray, device: str):
    """Build multi-world Newton model from exported USD.

    Each world is placed at its recorded root body position so that
    broad-phase collision covers the correct terrain patch.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    num_worlds = world_positions.shape[0]

    stage = Usd.Stage.Open(usd_path)
    layer = stage.GetRootLayer()
    for sp in ("/World", "/World/envs"):
        spec = layer.GetPrimAtPath(sp)
        if spec and spec.specifier == Sdf.SpecifierOver:
            spec.specifier = Sdf.SpecifierDef

    env_root = None
    def _v(path):
        nonlocal env_root
        if env_root is None and re.match(r"/World/envs/(env_\d+)$", str(path)):
            env_root = str(path)
    layer.Traverse(Sdf.Path.absoluteRootPath, _v)

    up_axis = UsdGeom.GetStageUpAxis(stage)
    resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    proto = ModelBuilder(up_axis=up_axis)
    if env_root:
        proto.add_usd(stage, root_path=env_root, schema_resolvers=resolvers,
                       load_visual_shapes=False)

    builder = ModelBuilder(up_axis=up_axis)
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim and world_prim.IsValid():
        for child in world_prim.GetChildren():
            cp = child.GetPath().pathString
            if cp.startswith("/World/envs"):
                continue
            for p in Usd.PrimRange(child):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    builder.add_usd(stage, root_path=cp, load_visual_shapes=False)
                    print(f"  Loaded collision: {cp}")
                    break

    for w in range(num_worlds):
        rx, ry = float(world_positions[w, 0]), float(world_positions[w, 1])
        builder.begin_world()
        builder.add_builder(proto, xform=wp.transform((rx, ry, 0.0), wp.quat_identity()))
        builder.end_world()

    model = builder.finalize(device=device)
    nw = getattr(model, "world_count", num_worlds)
    print(f"  Model: {model.body_count} bodies, {model.joint_count} joints, {nw} worlds")
    return model


def _set_state(model, state, jq: np.ndarray, jqd: np.ndarray,
               bq: np.ndarray | None, bqd: np.ndarray | None, device: str):
    """Write full state directly. No FK -- uses recorded body_q/body_qd."""
    jc_starts = model.joint_coord_world_start.numpy()
    jd_starts = model.joint_dof_world_start.numpy()
    nw = getattr(model, "world_count", 1) or 1
    jq_t = wp.to_torch(state.joint_q)
    jqd_t = wp.to_torch(state.joint_qd)
    total_jc = int(jc_starts[nw])
    total_jd = int(jd_starts[nw])

    if len(jq) == total_jc:
        jq_t.copy_(torch.tensor(jq, dtype=torch.float32, device=device))
        jqd_t.copy_(torch.tensor(jqd, dtype=torch.float32, device=device))
    else:
        jq_dev = torch.tensor(jq, dtype=torch.float32, device=device)
        jqd_dev = torch.tensor(jqd, dtype=torch.float32, device=device)
        for w in range(nw):
            jc0, jc1 = int(jc_starts[w]), int(jc_starts[w + 1])
            jd0, jd1 = int(jd_starts[w]), int(jd_starts[w + 1])
            if jc1 - jc0 == len(jq):
                jq_t[jc0:jc1] = jq_dev
            if jd1 - jd0 == len(jqd):
                jqd_t[jd0:jd1] = jqd_dev

    if bq is not None and state.body_q is not None:
        bq_flat = bq.reshape(-1) if bq.ndim > 1 else bq
        wp.to_torch(state.body_q).view(-1).copy_(
            torch.tensor(bq_flat, dtype=torch.float32, device=device))
    if bqd is not None and state.body_qd is not None:
        bqd_flat = bqd.reshape(-1) if bqd.ndim > 1 else bqd
        wp.to_torch(state.body_qd).view(-1).copy_(
            torch.tensor(bqd_flat, dtype=torch.float32, device=device))

    if bq is None:
        eval_fk(model, state.joint_q, state.joint_qd, state, None)


def _apply_forces(model, control, qfrc: np.ndarray, device: str):
    """Write forces to control.joint_f."""
    jd_starts = model.joint_dof_world_start.numpy()
    nw = getattr(model, "world_count", 1) or 1
    joint_f_t = wp.to_torch(control.joint_f)
    total_jd = int(jd_starts[nw])

    if qfrc.ndim == 2:
        nv = qfrc.shape[1]
        qfrc_dev = torch.tensor(qfrc, dtype=torch.float32, device=device)
        for w in range(min(nw, qfrc.shape[0])):
            jd0 = int(jd_starts[w])
            joint_f_t[jd0 : jd0 + nv] = qfrc_dev[w]
    elif len(qfrc) == total_jd:
        joint_f_t.copy_(torch.tensor(qfrc, dtype=torch.float32, device=device))
    else:
        qfrc_dev = torch.tensor(qfrc, dtype=torch.float32, device=device)
        for w in range(nw):
            jd0 = int(jd_starts[w])
            joint_f_t[jd0 : jd0 + len(qfrc)] = qfrc_dev


def _read_config(ring) -> dict:
    """Read solver + collision config from cfg_* keys in the npz."""
    _INTEGRATOR_MAP = {0: "euler", 1: "implicit", 2: "implicit", 3: "implicitfast"}
    _CONE_MAP = {0: "pyramidal", 1: "elliptic"}
    cfg = {}
    for k in ring.files:
        if k.startswith("cfg_"):
            v = ring[k]
            cfg[k[4:]] = v.item() if v.ndim == 0 or (v.ndim == 1 and v.shape[0] == 1) else v[0]
    if "integrator" in cfg and isinstance(cfg["integrator"], (int, float, np.integer)):
        cfg["integrator"] = _INTEGRATOR_MAP.get(int(cfg["integrator"]), "implicitfast")
    if "cone" in cfg and isinstance(cfg["cone"], (int, float, np.integer)):
        cfg["cone"] = _CONE_MAP.get(int(cfg["cone"]), "pyramidal")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Reproduce MuJoCo Warp solver NaN.")
    parser.add_argument("npz_path", type=str, help="Ring buffer npz from NaN debug export")
    parser.add_argument("--nan_only", action="store_true",
                        help="Load only the NaN'd env (1 world).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    # --- Load data ---
    ring = np.load(args.npz_path, allow_pickle=True)
    n_frames = int(ring.get("buffer_size", ring["body_q"].shape[0]))
    env_ids = ring.get("exported_env_ids", np.array([0]))
    bad_world = int(env_ids[0])

    first_valid, last_valid = _find_valid_range(ring["body_q"], bad_world, bodies_per_env=17)
    nan_frame = min(last_valid + 1, n_frames - 1)

    jq_all = ring["joint_q"]      # (n_frames, total_jc)
    jqd_all = ring["joint_qd"]    # (n_frames, total_jd)
    bq_all = ring["body_q"]       # (n_frames, total_bodies, 7)
    bqd_all = ring.get("body_qd") # (n_frames, total_bodies, 6) or None

    force_history = ring.get("diag_qfrc_applied")  # (n_frames, num_worlds, nv) or None

    # Detect world layout
    coords_per_env = 19
    dofs_per_env = 18
    export_worlds = max(1, jq_all.shape[-1] // coords_per_env)
    is_full = export_worlds > 1
    bodies_per_env = bq_all.shape[1] // export_worlds if is_full else bq_all.shape[1]

    # Per-world root positions for model building (from first valid frame)
    if is_full:
        all_root_pos = bq_all[first_valid, ::bodies_per_env, :3]
    else:
        all_root_pos = bq_all[first_valid, 0:1, :3]

    # Camera target
    if is_full:
        robot_pos = bq_all[last_valid, bad_world * bodies_per_env, :3]
    else:
        robot_pos = bq_all[last_valid, 0, :3]

    # World count and data selection
    if args.nan_only:
        num_worlds = 1
        world_positions = all_root_pos[bad_world : bad_world + 1] if is_full else all_root_pos
        mode = f"NaN'd env {bad_world} only"
    else:
        num_worlds = export_worlds
        world_positions = all_root_pos
        mode = f"all {num_worlds} envs"

    # Read config
    cfg = _read_config(ring)
    print(f"  Failing env: {bad_world}")
    print(f"  Ring: {n_frames} frames, valid [{first_valid}, {last_valid}], NaN at {nan_frame}")
    print(f"  Mode: {mode}")
    print(f"  Config: {cfg}")

    dt = float(cfg.pop("dt", 1.0 / 200.0))
    num_substeps = int(cfg.pop("num_substeps", 1))
    max_triangle_pairs = cfg.pop("max_triangle_pairs", None)
    if max_triangle_pairs is not None:
        max_triangle_pairs = int(max_triangle_pairs)

    # Remaining cfg keys are solver kwargs
    solver_kwargs = cfg

    # --- Build model ---
    usd_path = args.npz_path.replace(".npz", ".usd")
    print(f"\nBuilding {num_worlds} worlds from {usd_path}...")
    model = _build_model(usd_path, world_positions, args.device)
    solver = SolverMuJoCo(model, **solver_kwargs)

    pipeline_kwargs = {}
    if max_triangle_pairs is not None:
        pipeline_kwargs["max_triangle_pairs"] = max_triangle_pairs
    pipeline = CollisionPipeline(model, **pipeline_kwargs)
    contacts = pipeline.contacts()
    state = model.state()
    control = model.control()

    # --- Viewer ---
    viewer = None
    if args.visualize:
        from newton.viewer import ViewerGL
        viewer = ViewerGL(width=1920, height=1080, headless=False)
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        viewer.up_axis = 2
        viewer._render_left_panel = lambda: None
        viewer.renderer.draw_wireframe = True
        viewer.camera.pos = wp.vec3(robot_pos[0] + 3.0, robot_pos[1] - 3.0, 2.0)
        viewer.camera.yaw, viewer.camera.pitch = 90.0, -15.0

    # --- Load pre-NaN mjw_data if available ---
    pre_nan_mjw_path = args.npz_path.replace(".npz", "_pre_nan_mjw.npz")
    pre_nan_mjw = None
    try:
        pre_nan_mjw = np.load(pre_nan_mjw_path, allow_pickle=True)
        print(f"  Loaded pre-NaN mjw_data: {sorted(pre_nan_mjw.files)}")
    except FileNotFoundError:
        print(f"  No pre-NaN mjw_data found ({pre_nan_mjw_path})")

    # --- Replay: reset state each step, apply recorded forces, step once ---
    print(f"\nReplaying frames {first_valid}..{nan_frame} ({nan_frame - first_valid + 1} steps)...")
    sim_time = 0.0
    max_iter = solver_kwargs["iterations"]

    for frame in range(first_valid, nan_frame + 1):
        if viewer is not None and not viewer.is_running():
            break

        # Get state and forces for this frame
        if args.nan_only and is_full:
            w = bad_world
            jq = jq_all[frame, w * coords_per_env : (w + 1) * coords_per_env]
            jqd = jqd_all[frame, w * dofs_per_env : (w + 1) * dofs_per_env]
            bq = bq_all[frame, w * bodies_per_env : (w + 1) * bodies_per_env]
            bqd = bqd_all[frame, w * bodies_per_env : (w + 1) * bodies_per_env] if bqd_all is not None else None
        else:
            jq = jq_all[frame]
            jqd = jqd_all[frame]
            bq = bq_all[frame]
            bqd = bqd_all[frame] if bqd_all is not None else None

        if not np.isfinite(jq).all() or not np.isfinite(bq).all():
            print(f"  Frame {frame}: state has NaN/Inf, skipping")
            continue

        # Reset to exact recorded state (no FK recomputation)
        _set_state(model, state, jq, jqd, bq, bqd, args.device)

        # Apply recorded forces
        if force_history is not None:
            forces = force_history[frame]
            if args.nan_only and forces.ndim == 2 and is_full:
                forces = forces[bad_world]
            _apply_forces(model, control, forces, args.device)

        # On the last valid frame, inject pre-NaN mjw_data to reproduce warm-start
        if frame == nan_frame - 1 and pre_nan_mjw is not None:
            # Inject pre-NaN mjw_data and call solver internals directly.
            # This bypasses collide() and _convert_contacts which would
            # overwrite the recorded contact state.
            mjd = solver.mjw_data
            injected = 0
            for k in pre_nan_mjw.files:
                if k.startswith("efc_") or k.startswith("contact_"):
                    prefix, attr = k.split("_", 1)
                    sub = getattr(mjd, prefix, None)
                    if sub is not None:
                        target = getattr(sub, attr, None)
                        if target is not None and isinstance(target, wp.array):
                            src = pre_nan_mjw[k]
                            if target.shape == src.shape:
                                target.assign(wp.array(src, dtype=target.dtype, device=args.device))
                                injected += 1
                else:
                    target = getattr(mjd, k, None)
                    if target is not None and isinstance(target, wp.array):
                        src = pre_nan_mjw[k]
                        if target.shape == src.shape:
                            target.assign(wp.array(src, dtype=target.dtype, device=args.device))
                            injected += 1
            print(f"  Injected {injected} pre-NaN arrays, calling _mujoco_warp_step directly")
            solver._mujoco_warp_step()
            solver._update_newton_state(model, state, mjd, state_prev=state)
        else:
            # Normal step: collide → solve → clear
            pipeline.collide(state, contacts)
            for _sub in range(num_substeps):
                solver.step(state, state, control, contacts, dt)
                state.clear_forces()
        sim_time += dt * num_substeps

        # Check result
        niter = wp.to_torch(solver.mjw_data.solver_niter)
        body_q = wp.to_torch(state.body_q)
        has_nan = bool(body_q.isnan().any().item())
        max_niter = int(niter.max().item())

        if viewer is not None:
            viewer.begin_frame(sim_time)
            viewer.log_state(state)
            viewer.end_frame()
            _time.sleep(dt * 0.5)

        if has_nan:
            nan_worlds = torch.where(body_q.isnan().any(dim=-1))[0].tolist()
            nan_world_ids = [b // bodies_per_env for b in nan_worlds[:5]]
            print(f"\n*** NaN REPRODUCED at frame {frame} ***")
            print(f"  niter={max_niter}, NaN in worlds: {nan_world_ids}")
            if viewer is not None:
                while viewer.is_running():
                    viewer.begin_frame(sim_time)
                    viewer.log_state(state)
                    viewer.end_frame()
                    _time.sleep(0.016)
            sys.exit(0)

        if frame % 10 == 0 or max_niter >= max_iter:
            print(f"  Frame {frame}: niter={max_niter}, ok")

    print(f"\nCompleted replay. Looping from start...")
    while True:
        for frame in range(first_valid, nan_frame):
            if viewer is not None and not viewer.is_running():
                sys.exit(0)

            if args.nan_only and is_full:
                w = bad_world
                bq = bq_all[frame, w * bodies_per_env : (w + 1) * bodies_per_env]
            else:
                bq = bq_all[frame]

            if not np.isfinite(bq).all():
                continue

            bq_flat = bq.reshape(-1)
            wp.to_torch(state.body_q).view(-1).copy_(
                torch.tensor(bq_flat, dtype=torch.float32, device=args.device))
            sim_time += dt * num_substeps

            if viewer is not None:
                viewer.begin_frame(sim_time)
                viewer.log_state(state)
                viewer.end_frame()
                _time.sleep(dt * 0.5)


if __name__ == "__main__":
    main()
