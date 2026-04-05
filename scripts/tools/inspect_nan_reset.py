# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the exact robot state at NaN frame 0 (post-reset, pre-step).

Loads one or more NaN debug npz files, extracts the last valid frame
(the state written by reset, just before the solver step that NaN'd),
builds a multi-world Newton model with one world per NaN'd env, and
displays them side by side in the viewer.

Usage:
    python scripts/tools/inspect_nan_reset.py nan_debug/nan_replay_*.npz
    python scripts/tools/inspect_nan_reset.py nan_debug/nan_replay_A.npz nan_debug/nan_replay_B.npz
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
from newton.viewer import ViewerGL


def _extract_env_state(npz_path: str):
    """Extract the post-reset (pre-step) state for the NaN'd env from an npz.

    Prefers ``pre_step_*`` arrays (the exact solver input that caused NaN).
    Falls back to the last valid ring frame if ``pre_step_*`` is absent.
    """
    d = np.load(npz_path, allow_pickle=True)
    eid = int(d["exported_env_ids"][0])
    wc = int(d["world_count"])
    jq_all = d["joint_q"]
    jqd_all = d["joint_qd"]
    bq_all = d["body_q"]
    bqd_all = d.get("body_qd")

    jc_per = jq_all.shape[1] // wc
    jd_per = jqd_all.shape[1] // wc
    bodies_per = bq_all.shape[1] // wc

    # Prefer pre_step snapshot (exact post-reset state before solver NaN'd)
    has_pre_step = "pre_step_joint_q" in d
    if has_pre_step:
        ps_jq = d["pre_step_joint_q"]
        ps_jqd = d["pre_step_joint_qd"]
        ps_bq = d.get("pre_step_body_q")
        ps_bqd = d.get("pre_step_body_qd")

        jq = ps_jq[eid * jc_per : (eid + 1) * jc_per]
        jqd = ps_jqd[eid * jd_per : (eid + 1) * jd_per]
        bq = ps_bq.reshape(-1, 7)[eid * bodies_per : (eid + 1) * bodies_per] if ps_bq is not None else None
        bqd = ps_bqd.reshape(-1, 6)[eid * bodies_per : (eid + 1) * bodies_per] if ps_bqd is not None else None
        source = "pre_step (post-reset)"
    else:
        # Fallback: last valid ring frame (pre-reset, not ideal)
        for f in range(jq_all.shape[0] - 1, -1, -1):
            q = jq_all[f, eid * jc_per : (eid + 1) * jc_per]
            if np.isfinite(q).all():
                break
        jq = jq_all[f, eid * jc_per : (eid + 1) * jc_per]
        jqd = jqd_all[f, eid * jd_per : (eid + 1) * jd_per]
        bq = bq_all[f, eid * bodies_per : (eid + 1) * bodies_per]
        bqd = bqd_all[f, eid * bodies_per : (eid + 1) * bodies_per] if bqd_all is not None else None
        source = f"ring frame {f} (pre-reset fallback)"

    # NaN frame forces
    forces = None
    fh = d.get("diag_qfrc_applied")
    for f in range(jq_all.shape[0] - 1, -1, -1):
        q = jq_all[f, eid * jc_per : (eid + 1) * jc_per]
        if np.isfinite(q).all():
            break
    nan_frame = min(f + 1, jq_all.shape[0] - 1)
    if fh is not None and fh.ndim == 3:
        forces = fh[nan_frame, eid]

    root_pos = jq[:3] if np.isfinite(jq[:3]).all() else (bq[0, :3] if bq is not None else np.zeros(3))
    cdist = d["diag_contact_dist_min"][nan_frame, eid] if "diag_contact_dist_min" in d else None
    niter = d["diag_solver_niter"][nan_frame, eid] if "diag_solver_niter" in d else None
    ep_step = None
    if "episode_length_buf" in d:
        ep_step = int(d["episode_length_buf"][eid])

    cfg = {}
    for k in d.files:
        if k.startswith("cfg_"):
            v = d[k]
            cfg[k[4:]] = v.item() if v.ndim == 0 or (v.ndim == 1 and v.shape[0] == 1) else v[0]

    return {
        "env_id": eid,
        "source": source,
        "jq": jq,
        "jqd": jqd,
        "bq": bq,
        "bqd": bqd,
        "root_pos": root_pos,
        "forces": forces,
        "contact_dist_min": cdist,
        "solver_niter": niter,
        "episode_step": ep_step,
        "jc_per": jc_per,
        "jd_per": jd_per,
        "bodies_per": bodies_per,
        "cfg": cfg,
        "npz_path": npz_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect NaN reset states visually.")
    parser.add_argument("npz_paths", nargs="+", help="NaN debug npz files")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--step", action="store_true",
                        help="After showing pre-NaN state, step physics once to reproduce")
    args = parser.parse_args()

    envs = [_extract_env_state(p) for p in args.npz_paths]
    num_worlds = len(envs)

    for i, e in enumerate(envs):
        print(f"  [{i}] env {e['env_id']} from {e['npz_path']}")
        print(f"      source: {e['source']}")
        print(f"      episode_step={e['episode_step']}, contact_dist={e['contact_dist_min']}, niter={e['solver_niter']}")
        print(f"      root_pos={e['root_pos']}")
        print(f"      root_quat={e['jq'][3:7]}")
        print(f"      joint_pos={e['jq'][7:]}")

    # Build model: one world per NaN'd env, each at its root position
    usd_path = args.npz_paths[0].replace(".npz", ".usd")
    print(f"\nBuilding {num_worlds} worlds from {usd_path}...")

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

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
    SolverMuJoCo.register_custom_attributes(proto)
    if env_root:
        proto.add_usd(stage, root_path=env_root, schema_resolvers=resolvers,
                       load_visual_shapes=True, skip_mesh_approximation=True)
        proto.approximate_meshes("convex_hull", keep_visual_shapes=True)

    builder = ModelBuilder(up_axis=up_axis)
    # Load terrain/global collision geometry
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim and world_prim.IsValid():
        for child in world_prim.GetChildren():
            cp = child.GetPath().pathString
            if cp.startswith("/World/envs"):
                continue
            for p in Usd.PrimRange(child):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    builder.add_usd(stage, root_path=cp, load_visual_shapes=True,
                                    schema_resolvers=resolvers)
                    print(f"  Loaded collision: {cp}")
                    break

    # Spacing between displayed worlds
    spacing = 5.0
    for i, e in enumerate(envs):
        rx, ry = float(e["root_pos"][0]), float(e["root_pos"][1])
        builder.begin_world()
        builder.add_builder(proto, xform=wp.transform((rx, ry, 0.0), wp.quat_identity()))
        builder.end_world()

    model = builder.finalize(device=args.device)
    state = model.state()
    control = model.control()

    # Read solver config from first env
    cfg = dict(envs[0]["cfg"])
    _INTEGRATOR_MAP = {0: "euler", 1: "implicit", 2: "implicit", 3: "implicitfast"}
    _CONE_MAP = {0: "pyramidal", 1: "elliptic"}
    if "integrator" in cfg and isinstance(cfg["integrator"], (int, float, np.integer)):
        cfg["integrator"] = _INTEGRATOR_MAP.get(int(cfg["integrator"]), "implicitfast")
    if "cone" in cfg and isinstance(cfg["cone"], (int, float, np.integer)):
        cfg["cone"] = _CONE_MAP.get(int(cfg["cone"]), "pyramidal")
    dt = float(cfg.pop("dt", 0.005))
    cfg.pop("num_substeps", None)
    max_triangle_pairs = cfg.pop("max_triangle_pairs", None)

    solver = SolverMuJoCo(model, **cfg)

    pipeline_kwargs = {}
    if max_triangle_pairs is not None:
        pipeline_kwargs["max_triangle_pairs"] = int(max_triangle_pairs)
    pipeline = CollisionPipeline(model, **pipeline_kwargs)
    contacts = pipeline.contacts()

    print(f"  Model: {model.body_count} bodies, {model.joint_count} joints, {model.world_count} worlds")

    # Set each world to its recorded state
    jc_starts = model.joint_coord_world_start.numpy()
    jd_starts = model.joint_dof_world_start.numpy()

    for w, e in enumerate(envs):
        jc0 = int(jc_starts[w])
        jd0 = int(jd_starts[w])
        jq_t = wp.to_torch(state.joint_q)
        jqd_t = wp.to_torch(state.joint_qd)
        jq_t[jc0 : jc0 + e["jc_per"]] = torch.tensor(e["jq"], dtype=torch.float32, device=args.device)
        jqd_t[jd0 : jd0 + e["jd_per"]] = torch.tensor(e["jqd"], dtype=torch.float32, device=args.device)

    eval_fk(model, state.joint_q, state.joint_qd, state, None)

    # Viewer
    center = envs[0]["root_pos"]
    viewer = ViewerGL(width=1920, height=1080, headless=False)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    viewer.up_axis = 2
    viewer.camera.pos = wp.vec3(center[0] + 3.0, center[1] - 3.0, 2.0)
    viewer.camera.yaw, viewer.camera.pitch = 90.0, -15.0

    print(f"\nDisplaying {num_worlds} robot(s) at their pre-NaN (post-reset) state.")
    print("Close the viewer to exit, or use --step to step physics.\n")

    if args.step:
        print("Press Enter to step physics (reproduce the NaN)...")
        # Show initial state first
        for _ in range(60):
            if not viewer.is_running():
                sys.exit(0)
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()
            _time.sleep(1.0 / 60.0)

        # Apply forces and step
        for w, e in enumerate(envs):
            if e["forces"] is not None:
                jd0 = int(jd_starts[w])
                jf = wp.to_torch(control.joint_f)
                jf[jd0 : jd0 + e["jd_per"]] = torch.tensor(
                    e["forces"], dtype=torch.float32, device=args.device
                )

        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, dt)
        state.clear_forces()
        eval_fk(model, state.joint_q, state.joint_qd, state, None)

        has_nan = bool(wp.to_torch(state.body_q).isnan().any().item())
        niter = int(wp.to_torch(solver.mjw_data.solver_niter).max().item())
        print(f"After step: has_nan={has_nan}, max_solver_niter={niter}")

    sim_time = 0.0
    while viewer.is_running():
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
        _time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
