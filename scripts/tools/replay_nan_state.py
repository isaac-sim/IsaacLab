# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and replay NaN state snapshots exported by the NewtonManager debug buffer.

Usage:
    # Print summary only
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /path/to/nan_replay.npz

    # Print summary + matplotlib plots
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /path/to/nan_replay.npz --plot

    # Replay in Newton viewer (auto-discovers .usd next to .npz)
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /path/to/nan_replay.npz --visualizer newton

    # Replay at 0.25x speed
    ./isaaclab.sh -p scripts/tools/replay_nan_state.py /path/to/nan_replay.npz --visualizer newton --speed 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

def _print_summary(npz_path: str) -> None:
    """Print a text summary of the npz contents."""
    

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {npz_path}: {e}", file=sys.stderr)
        sys.exit(1)

    n = int(data.get("buffer_size", data["joint_q"].shape[0] if "joint_q" in data else 0))
    if n == 0 and "joint_q" in data:
        n = data["joint_q"].shape[0]
    sim_time = float(data.get("sim_time", 0.0))

    print("=== NaN replay summary ===")
    print(f"File: {npz_path}")
    print(f"Buffer size (steps): {n}")
    print(f"Sim time at export: {sim_time}")
    if "exported_env_ids" in data:
        env_ids = data["exported_env_ids"]
        print(f"Exported env(s) only: {env_ids.tolist()}")

    usd_path = _find_usd_path(npz_path)
    if usd_path:
        print(f"Scene USD: {usd_path}")

    for key in ("body_q", "body_qd", "joint_q", "joint_qd"):
        if key not in data:
            continue
        arr = data[key]
        print(f"\n{key}: shape {arr.shape}, dtype {arr.dtype}")
        nan_per_step = np.isnan(arr).reshape(arr.shape[0], -1).any(axis=1)
        first_nan = np.where(nan_per_step)[0]
        if len(first_nan) > 0:
            print(f"  First step with NaN: {int(first_nan[0])} (last step {n-1} is the incident)")
        print(f"  Min: {np.nanmin(arr):.6g}, Max: {np.nanmax(arr):.6g}")

    # Print per-body and per-joint details at the last valid step (one before NaN)
    last_valid = n - 2 if n >= 2 else 0
    print(f"\n--- Per-body/joint state at step {last_valid} (last valid before NaN) ---")

    if "body_qd" in data:
        bqd = data["body_qd"]
        if last_valid < bqd.shape[0]:
            frame = bqd[last_valid]
            print(f"\nbody_qd (link velocities) [{frame.shape[0]} bodies]:")
            print(f"  {'Body':>4}  {'lin_x':>10} {'lin_y':>10} {'lin_z':>10}  {'ang_x':>10} {'ang_y':>10} {'ang_z':>10}")
            for bi in range(frame.shape[0]):
                v = frame[bi]
                print(f"  {bi:4d}  {v[0]:10.4f} {v[1]:10.4f} {v[2]:10.4f}  {v[3]:10.4f} {v[4]:10.4f} {v[5]:10.4f}")

    if "joint_qd" in data:
        jqd = data["joint_qd"]
        if last_valid < jqd.shape[0]:
            frame = jqd[last_valid]
            print(f"\njoint_qd (joint velocities) [{frame.shape[0]} dofs]:")
            for ji in range(frame.shape[0]):
                print(f"  dof {ji:3d}: {frame[ji]:12.4f}")

    if "joint_q" in data:
        jq = data["joint_q"]
        if last_valid < jq.shape[0]:
            frame = jq[last_valid]
            print(f"\njoint_q (joint positions) [{frame.shape[0]} coords]:")
            for ji in range(frame.shape[0]):
                print(f"  coord {ji:3d}: {frame[ji]:12.6f}")

    # --- Diagnostics summary ---
    diag_keys = [k for k in data.files if k.startswith("diag_")] if hasattr(data, "files") else [k for k in data if k.startswith("diag_")]
    if diag_keys:
        print(f"\n{'='*70}")
        print("  DIAGNOSTICS")
        print(f"{'='*70}")

        _print_diag_timeline(data, n, last_valid)
        _print_root_cause_analysis(data, n, last_valid)


def _print_diag_timeline(data, n: int, last_valid: int) -> None:
    """Print per-step diagnostic summary for the last few steps before NaN."""
    window = min(10, last_valid + 1)
    start = last_valid - window + 1

    print(f"\n  Timeline (steps {start}..{last_valid}, last {window} before NaN):")
    header = f"  {'Step':>5}"
    has_niter = "diag_solver_niter" in data
    has_qfrc = "diag_qfrc_constraint" in data
    has_torque = "diag_qfrc_actuator" in data
    has_qacc = "diag_qacc" in data
    has_qm = "diag_qM_diag_min" in data
    has_cdist = "diag_contact_dist_min" in data

    if has_niter:
        header += f"  {'iters':>6}"
    if has_qfrc:
        header += f"  {'|F_con|':>10}"
    if has_torque:
        header += f"  {'|torque|':>10}"
    if has_qacc:
        header += f"  {'|accel|':>10}"
    if has_qm:
        header += f"  {'min_qM':>10}"
    if has_cdist:
        header += f"  {'penetr':>10}"
    print(header)

    for step in range(start, last_valid + 1):
        row = f"  {step:5d}"
        if has_niter:
            val = data["diag_solver_niter"][step]
            v = int(val) if val.ndim == 0 else int(val.max())
            row += f"  {v:6d}"
        if has_qfrc:
            val = data["diag_qfrc_constraint"][step]
            row += f"  {np.abs(val).max():10.2f}"
        if has_torque:
            val = data["diag_qfrc_actuator"][step]
            row += f"  {np.abs(val).max():10.2f}"
        if has_qacc:
            val = data["diag_qacc"][step]
            row += f"  {np.abs(val).max():10.2f}"
        if has_qm:
            val = data["diag_qM_diag_min"][step]
            v = float(val) if val.ndim == 0 else float(val.min())
            row += f"  {v:10.6f}"
        if has_cdist:
            val = data["diag_contact_dist_min"][step]
            v = float(val) if val.ndim == 0 else float(val.min())
            row += f"  {v:10.6f}"
        print(row)


def _print_root_cause_analysis(data, n: int, last_valid: int) -> None:
    """Auto-flag likely root cause based on diagnostic patterns."""
    print(f"\n  ROOT CAUSE ANALYSIS:")
    flags = []

    if "diag_solver_niter" in data and last_valid >= 0:
        niter = data["diag_solver_niter"]
        max_iters = int(niter[last_valid].max()) if niter[last_valid].ndim > 0 else int(niter[last_valid])
        if max_iters >= 95:
            flags.append(f"  >> SOLVER NON-CONVERGENCE: solver used {max_iters} iterations at step {last_valid}")

    if "diag_qM_diag_min" in data and last_valid >= 0:
        qm = data["diag_qM_diag_min"]
        min_qm = float(qm[last_valid].min()) if qm[last_valid].ndim > 0 else float(qm[last_valid])
        if min_qm < 1e-6:
            flags.append(f"  >> NEAR-SINGULAR MASS MATRIX: min(diag(qM)) = {min_qm:.2e} at step {last_valid}")

    if "diag_contact_dist_min" in data and last_valid >= 0:
        cdist = data["diag_contact_dist_min"]
        min_dist = float(cdist[last_valid].min()) if cdist[last_valid].ndim > 0 else float(cdist[last_valid])
        if min_dist < -0.01:
            flags.append(f"  >> DEEP PENETRATION: contact dist = {min_dist:.4f} m at step {last_valid}")

    if "diag_qfrc_constraint" in data and last_valid >= 0:
        frc = data["diag_qfrc_constraint"]
        max_frc = float(np.abs(frc[last_valid]).max())
        if max_frc > 1e4:
            flags.append(f"  >> EXTREME CONTACT FORCE: |qfrc_constraint| = {max_frc:.1f} at step {last_valid}")

    if "diag_qfrc_actuator" in data and last_valid >= 0:
        trq = data["diag_qfrc_actuator"]
        max_trq = float(np.abs(trq[last_valid]).max())
        if max_trq > 1e4:
            flags.append(f"  >> EXTREME ACTUATOR TORQUE: |qfrc_actuator| = {max_trq:.1f} at step {last_valid}")

    if "diag_qacc" in data and last_valid >= 0:
        acc = data["diag_qacc"]
        max_acc = float(np.abs(acc[last_valid]).max())
        if max_acc > 1e6:
            has_normal_forces = True
            if "diag_qfrc_constraint" in data:
                has_normal_forces = float(np.abs(data["diag_qfrc_constraint"][last_valid]).max()) < 1e4
            if has_normal_forces:
                flags.append(f"  >> ACCELERATION SPIKE (forces normal): |qacc| = {max_acc:.1f} => likely inv(qM) blowup")

    # Teleport detection from body_q
    if "body_q" in data and last_valid >= 1:
        bq = data["body_q"]
        pos_curr = bq[last_valid, :, :3]
        pos_prev = bq[last_valid - 1, :, :3]
        jump = np.linalg.norm(pos_curr - pos_prev, axis=-1).max()
        if jump > 1.0:
            worst = int(np.argmax(np.linalg.norm(pos_curr - pos_prev, axis=-1)))
            flags.append(f"  >> POSITION TELEPORT: body {worst} jumped {jump:.2f} m between steps {last_valid-1}->{last_valid}")

    if flags:
        for f in flags:
            print(f)
    else:
        print("  No obvious root cause detected from diagnostics.")


def _plot(npz_path: str, output_dir: str | None) -> None:
    """Generate matplotlib plots of the state trajectories."""
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    n = int(data.get("buffer_size", data["joint_q"].shape[0] if "joint_q" in data else 0))

    try:
        import matplotlib

        if output_dir:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping --plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(n)

    if "body_q" in data:
        bq = data["body_q"]
        if bq.ndim == 3:
            pos = bq[:, 0, :3]
        elif bq.ndim == 2 and bq.shape[1] >= 3:
            pos = bq[:, :3]
        else:
            pos = None
        if pos is not None:
            for i, label in enumerate("xyz"):
                axes[0].plot(t, pos[:, i], label=label)
            axes[0].set_ylabel("Body 0 position [m]")
            axes[0].legend(loc="upper right")
            axes[0].set_title("First body position (world)")
            axes[0].grid(True, alpha=0.3)

    if "joint_qd" in data:
        jqd = data["joint_qd"]
        ndof = min(3, jqd.shape[1])
        for i in range(ndof):
            axes[1].plot(t, jqd[:, i], label=f"qd[{i}]")
        axes[1].set_ylabel("Joint velocity")
        axes[1].set_xlabel("Step (last = NaN incident)")
        axes[1].legend(loc="upper right")
        axes[1].set_title("First 3 joint velocities")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "nan_replay_plot.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    else:
        plt.show()


def _find_usd_path(npz_path: str) -> str | None:
    """Find the companion .usd file for a .npz export."""
    base = os.path.splitext(npz_path)[0]
    for ext in (".usd", ".usda", ".usdc"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _prepare_stage(usd_path: str):
    """Open the exported USD and fix ``over`` ancestors so the stage composes.

    The NaN exporter uses ``Sdf.CopySpec`` which preserves the original
    specifier.  Ancestor prims (``/World``, ``/World/envs``) are stored as
    ``over`` because they were overrides on the live stage.  Converting them
    to ``def`` lets USD compose and traverse the env subtree.

    Returns:
        (stage, env_root_path) — the opened stage and the path to the
        environment prim (e.g. ``/World/envs/env_1303``).
    """
    import re

    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(usd_path)
    layer = stage.GetRootLayer()

    for ancestor in ("/World", "/World/envs"):
        spec = layer.GetPrimAtPath(ancestor)
        if spec and spec.specifier == Sdf.SpecifierOver:
            spec.specifier = Sdf.SpecifierDef

    env_re = re.compile(r"/World/envs/(env_\d+)$")
    env_root = None

    def _visitor(path):
        nonlocal env_root
        if env_root is None and env_re.match(str(path)):
            env_root = str(path)

    layer.Traverse(Sdf.Path.absoluteRootPath, _visitor)
    return stage, env_root


def _replay_newton(npz_path: str, speed: float, loop: bool) -> None:
    """Replay state snapshots in the Newton ViewerGL.

    Builds a Newton Model from the exported USD scene, creates a State, then
    steps through each recorded snapshot writing body_q/body_qd/joint_q/joint_qd
    into the state and rendering each frame.
    """
    import numpy as np
    import warp as wp
    from newton import ModelBuilder
    from newton.viewer import ViewerGL
    from pxr import Usd, UsdGeom, UsdPhysics

    data = np.load(npz_path, allow_pickle=True)
    n_steps = int(data.get("buffer_size", 0))
    if n_steps == 0 and "joint_q" in data:
        n_steps = data["joint_q"].shape[0]

    usd_path = _find_usd_path(npz_path)
    if usd_path is None:
        print(
            "No companion .usd file found next to the .npz.\n"
            "The Newton viewer requires the exported scene USD to build a Model.\n"
            "Re-run training with the NaN replay buffer enabled to export both .npz and .usd.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building Newton model from: {usd_path}")
    stage, env_root = _prepare_stage(usd_path)
    up_axis = UsdGeom.GetStageUpAxis(stage)

    builder = ModelBuilder(up_axis=up_axis)
    if env_root:
        builder.add_usd(stage, root_path=env_root, load_visual_shapes=False)
    else:
        builder.add_usd(stage, load_visual_shapes=False)
    # Load any global collision geometry (terrain, ground planes, etc.)
    # that lives under /World but outside /World/envs.
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim and world_prim.IsValid():
        for child in world_prim.GetChildren():
            child_path = child.GetPath().pathString
            if child_path.startswith("/World/envs"):
                continue
            has_collision = any(
                p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(child)
            )
            if has_collision:
                builder.add_usd(stage, root_path=child_path, load_visual_shapes=True)
                print(f"Loaded collision geometry from {child_path}")
    model = builder.finalize(device="cpu")
    state = model.state()

    if model.body_count == 0:
        print("Warning: model has 0 bodies. The USD may not contain resolvable physics prims.", file=sys.stderr)

    body_q_all = data.get("body_q")
    body_qd_all = data.get("body_qd")
    joint_q_all = data.get("joint_q")
    joint_qd_all = data.get("joint_qd")
    sim_time_at_export = float(data.get("sim_time", 0.0))

    dt = 1.0 / 60.0
    env_ids = data["exported_env_ids"].tolist() if "exported_env_ids" in data else []
    print(f"Model: {model.body_count} bodies, {model.joint_count} joints")

    viewer = ViewerGL(width=1920, height=1080, headless=False)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    viewer.up_axis = 2  # Z-up

    # Work around Newton ViewerGL imgui color_edit3 incompatibility (expects
    # ImVec4 but receives a plain tuple).  Patch before the first frame so we
    # never leave imgui in a half-finished state.
    viewer._render_left_panel = lambda: None

    # Enable wireframe rendering to see mesh triangle structure
    viewer.renderer.draw_wireframe = True

    # Point camera at the robot using the first frame's body positions.
    if body_q_all is not None and body_q_all.shape[0] > 0:
        first_frame = body_q_all[0]
        valid_mask = np.isfinite(first_frame).all(axis=-1)
        if valid_mask.any():
            positions = first_frame[valid_mask, :3]
            center = positions.mean(axis=0)
            extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
            cam_dist = max(extent * 2.0, 1.5)
            viewer.camera.pos = wp.vec3(
                float(center[0]) + cam_dist * 0.5,
                float(center[1]) - cam_dist * 0.5,
                float(center[2]) + cam_dist * 0.4,
            )
            direction = np.array([center[0], center[1], center[2]]) - np.array(
                [float(viewer.camera.pos[0]), float(viewer.camera.pos[1]), float(viewer.camera.pos[2])]
            )
            viewer.camera.yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
            horiz = float(np.sqrt(direction[0] ** 2 + direction[1] ** 2))
            viewer.camera.pitch = float(np.degrees(np.arctan2(direction[2], horiz)))
            print(f"Camera target: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

    print(f"Replaying {n_steps} snapshots at {speed}x speed (loop={loop})")
    if env_ids:
        print(f"Exported env_id(s): {env_ids}")
    print("Press ESC in the viewer window to exit.")

    frame_delay = dt / speed
    sim_time = sim_time_at_export - n_steps * dt

    def _assign_if_compatible(target, frame_data, dtype):
        """Assign frame data to a warp state array if shapes are compatible."""
        if target is None or frame_data is None:
            return
        if frame_data.shape[0] == target.shape[0]:
            target.assign(wp.array(frame_data, dtype=dtype, device="cpu"))

    # Build short name tables from model labels
    body_names = [lbl.rsplit("/", 1)[-1] for lbl in (model.body_label if hasattr(model, "body_label") else [])]
    joint_names = [lbl.rsplit("/", 1)[-1] for lbl in (model.joint_label if hasattr(model, "joint_label") else [])]
    # joint_q has one coord per joint (may have >1 for free joints)
    # joint_qd has one dof per joint
    # For display, map dof index -> joint name (approximate: 1 dof per joint for revolute)

    def _short(name: str, maxlen: int = 14) -> str:
        return name[:maxlen].ljust(maxlen)

    # Preload diagnostic arrays from npz
    diag_niter = data.get("diag_solver_niter")
    diag_qfrc = data.get("diag_qfrc_constraint")
    diag_torque = data.get("diag_qfrc_actuator")
    diag_qacc = data.get("diag_qacc")
    diag_qm = data.get("diag_qM_diag_min")
    diag_cdist = data.get("diag_contact_dist_min")

    def _print_frame_state(idx, bq, bqd, jq, jqd):
        """Print full per-body/joint state + diagnostics for the current frame."""
        print(f"\n{'='*100}")
        print(f"  Step {idx}/{n_steps}")

        # Diagnostics header line
        parts = []
        if diag_niter is not None and idx < diag_niter.shape[0]:
            v = diag_niter[idx]
            iters = int(v) if v.ndim == 0 else int(v.max())
            warn = " !!" if iters >= 95 else ""
            parts.append(f"solver_iters={iters}{warn}")
        if diag_qfrc is not None and idx < diag_qfrc.shape[0]:
            parts.append(f"|F_con|={np.abs(diag_qfrc[idx]).max():.1f}")
        if diag_torque is not None and idx < diag_torque.shape[0]:
            parts.append(f"|torque|={np.abs(diag_torque[idx]).max():.1f}")
        if diag_qacc is not None and idx < diag_qacc.shape[0]:
            parts.append(f"|accel|={np.abs(diag_qacc[idx]).max():.1f}")
        if diag_qm is not None and idx < diag_qm.shape[0]:
            v = diag_qm[idx]
            qm_min = float(v) if v.ndim == 0 else float(v.min())
            warn = " !!" if qm_min < 1e-6 else ""
            parts.append(f"min_qM={qm_min:.6f}{warn}")
        if diag_cdist is not None and idx < diag_cdist.shape[0]:
            v = diag_cdist[idx]
            cd = float(v) if v.ndim == 0 else float(v.min())
            warn = " !!" if cd < -0.01 else ""
            parts.append(f"penetr={cd:.4f}{warn}")
        if parts:
            print(f"  {' | '.join(parts)}")

        print(f"{'='*100}")

        if bqd is not None:
            print(f"  {'Body':<14}  {'lin_x':>9} {'lin_y':>9} {'lin_z':>9}  {'ang_x':>9} {'ang_y':>9} {'ang_z':>9}")
            for bi in range(bqd.shape[0]):
                name = _short(body_names[bi]) if bi < len(body_names) else f"body_{bi:<8}"
                v = bqd[bi]
                print(f"  {name}  {v[0]:9.2f} {v[1]:9.2f} {v[2]:9.2f}  {v[3]:9.2f} {v[4]:9.2f} {v[5]:9.2f}")

        if jqd is not None:
            print(f"  {'Joint':<14}  {'vel':>12}  {'pos':>12}")
            for ji in range(jqd.shape[0]):
                name = _short(joint_names[ji]) if ji < len(joint_names) else f"dof_{ji:<9}"
                pos_val = f"{jq[ji]:12.4f}" if jq is not None and ji < jq.shape[0] else "         N/A"
                print(f"  {name}  {jqd[ji]:12.4f}  {pos_val}")

    running = True
    while running and viewer.is_running():
        for step_idx in range(n_steps):
            if not viewer.is_running():
                running = False
                break

            if body_q_all is not None and step_idx < body_q_all.shape[0]:
                _assign_if_compatible(state.body_q, body_q_all[step_idx], wp.transform)
            if body_qd_all is not None and step_idx < body_qd_all.shape[0]:
                _assign_if_compatible(state.body_qd, body_qd_all[step_idx], wp.spatial_vector)
            if joint_q_all is not None and step_idx < joint_q_all.shape[0]:
                _assign_if_compatible(state.joint_q, joint_q_all[step_idx], float)
            if joint_qd_all is not None and step_idx < joint_qd_all.shape[0]:
                _assign_if_compatible(state.joint_qd, joint_qd_all[step_idx], float)

            _print_frame_state(
                step_idx,
                body_q_all[step_idx] if body_q_all is not None and step_idx < body_q_all.shape[0] else None,
                body_qd_all[step_idx] if body_qd_all is not None and step_idx < body_qd_all.shape[0] else None,
                joint_q_all[step_idx] if joint_q_all is not None and step_idx < joint_q_all.shape[0] else None,
                joint_qd_all[step_idx] if joint_qd_all is not None and step_idx < joint_qd_all.shape[0] else None,
            )

            sim_time += dt
            viewer.begin_frame(sim_time)
            viewer.log_state(state)
            viewer.end_frame()

            time.sleep(frame_delay)

        if not loop:
            print("Replay complete. Close the viewer window to exit.")
            while viewer.is_running():
                viewer.begin_frame(sim_time)
                viewer.log_state(state)
                viewer.end_frame()
                time.sleep(frame_delay)
            break


def main():
    parser = argparse.ArgumentParser(
        description="Load NaN replay npz and print summary, plot, or replay in Newton viewer."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the nan_replay_*.npz file exported when NaN was detected.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot body_q positions (first body, xyz) and joint_qd (first 3 dof) over time.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plot images. If not set, plots are shown interactively.",
    )
    parser.add_argument(
        "--visualizer",
        type=str,
        default=None,
        choices=["newton"],
        help="Visualizer backend for replay. Currently supports: newton.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (e.g. 0.25 for quarter speed). Default: 1.0.",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Play the replay only once instead of looping.",
    )
    args = parser.parse_args()

    _print_summary(args.npz_path)

    if args.plot:
        _plot(args.npz_path, args.output_dir)

    if args.visualizer == "newton":
        _replay_newton(args.npz_path, speed=args.speed, loop=not args.no_loop)


if __name__ == "__main__":
    main()
