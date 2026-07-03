# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from the agile sysid stack (scripts/sys_id/fit.py), adapted from
# pace-sim2real (ETH Zurich RSL / NVIDIA Isaac).

"""CMA-ES system identification for implicit-actuator PD gains.

Replays a chirp trajectory recorded with the isaac_ros_sysid GUI through N
parallel envs and fits per-joint {stiffness, damping} of the ImplicitActuator
by minimising the position error between simulated and measured joints. Runs
on the Newton/mjwarp backend by default (the task's physics preset).

The dataset contract (keys, shapes, provenance rules) lives in
``data_contract.py`` and is fail-closed. Command shaping between the recorded
``des_dof_pos`` and the drive target is provenance-driven: an APPROXIMATE
reconstruction for the franka_fr3 ros2_control driver (clamp + EMA + Ruckig,
see ``fr3_target_shaping.py`` — unknown initial internal state and timestamp
alignment keep real-data fits gated), identity/ZOH otherwise. The loss masks
a heuristic burn-in window; shaper state is preserved, never sliced.

Usage
-----
fit:   python scripts/sysid/fit.py --data <run>/chirp_data.pt
eval:  python scripts/sysid/fit.py --data <heldout>/chirp_data.pt \\
           --eval_params logs/sysid/franka_fr3/<stamp>/best_candidate.pt
"""

# flake8: noqa: E402

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="CMA-ES sysid fitting for implicit-actuator gains.")
parser.add_argument("--num_envs", type=int, default=256, help="CMA-ES population size.")
parser.add_argument("--task", type=str, default="Isaac-Sysid-Franka-FR3-v0", help="Gym task name.")
parser.add_argument("--data", type=str, required=True, help="Path to .pt/.npz dataset (see data_contract.py).")
parser.add_argument(
    "--joint_order",
    nargs="+",
    default=None,
    help="Override the joints to fit. Default: `active_joint_names` from the data file.",
)
parser.add_argument(
    "--initial_mean_path",
    type=str,
    default=None,
    help="Path to a .pt warm-start mean (physical parameter space, shape (2N,)).",
)
parser.add_argument(
    "--warmstart_from_data",
    action="store_true",
    default=False,
    help="Seed the CMA-ES mean from the dataset's kp_used/kd_used metadata.",
)
parser.add_argument("--warmstart_sigma_scale", type=float, default=1.0, help="Sigma scale when warm-starting.")
parser.add_argument(
    "--eval_params",
    type=str,
    default=None,
    help=(
        "Evaluation-only mode: path to best_candidate.pt / mean_XXX.pt (or any (2N,) tensor). "
        "Rolls the parameters out on this dataset against the recorded-gain and asset-default "
        "baselines instead of fitting."
    ),
)
parser.add_argument(
    "--shaping",
    type=str,
    choices=["auto", "franka_fr3", "none"],
    default="auto",
    help=(
        "Command-shaping model between recorded des_dof_pos and the drive target. 'auto' "
        "(default) resolves the provenance stamped in the dataset and HARD-FAILS when absent "
        "or incomplete; forcing 'franka_fr3'/'none' is the explicit operator override."
    ),
)
parser.add_argument("--relative_dynamics", type=float, default=None, help="Ruckig limit scale override.")
parser.add_argument("--ema_alpha", type=float, default=None, help="Shaper EMA coefficient override.")
parser.add_argument("--shaper_rate_hz", type=float, default=None, help="Shaper/physics tick rate override (Hz).")
parser.add_argument(
    "--controller_update_rate",
    type=float,
    default=None,
    help="COMMAND-path rate (Hz) cross-check; default: controller_update_rate_hint, else the time axis.",
)
parser.add_argument(
    "--physics_rate",
    type=float,
    default=1000.0,
    help="Sim physics rate (Hz) when the dataset has no shaper rate (shaper_type none).",
)
parser.add_argument(
    "--loss_burn_in_s",
    type=float,
    default=None,
    help="Override the loss-masking burn-in window (s). Default: sized from the run's initial mismatch.",
)
parser.add_argument(
    "--allow_stale_fraction",
    type=float,
    default=0.0,
    help=(
        "Explicit unsafe override: accept datasets whose state_fresh mask marks up to this "
        "fraction of rows as stale (reused JointState). Stale rows are MASKED from the loss."
    ),
)
parser.add_argument(
    "--allow_missing_freshness",
    action="store_true",
    default=False,
    help="Explicit unsafe override: accept datasets without state_fresh/state_stamps (legacy data only).",
)
parser.add_argument(
    "--allow_truncated",
    action="store_true",
    default=False,
    help="Explicit unsafe override: accept safety-aborted/truncated runs (diagnostics only).",
)
parser.add_argument(
    "--eval_max_saturation",
    type=float,
    default=0.05,
    help="Eval verdict policy: fraction of (tick,joint) samples at the effort limit above which eval FAILS.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="Override cfg CMA-ES max iterations.")
parser.add_argument("--seed", type=int, default=0, help="CMA-ES seed (multi-seed runs are an acceptance gate).")
parser.add_argument("--log_dir", type=str, default=None, help="Log dir (default: logs/sysid/<robot_name>).")
parser.add_argument(
    "--plot_script",
    type=str,
    default=os.environ.get("SYSID_PLOT_CHIRP", ""),
    help="Optional isaac_ros_sysid plot_chirp.py for the post-fit overlay plot (default: $SYSID_PLOT_CHIRP).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  (registers tasks)
from isaaclab_tasks.contrib.sysid.config.franka_fr3.fr3_sysid_env_cfg import build_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cma_es import CMAESOptimizer  # noqa: E402
from data_contract import (  # noqa: E402
    build_loss_mask,
    canonical_indices,
    load_dataset,
    measure_convergence_burn_in,
    require_finite,
    resolve_shaper,
    validate_contract,
)


def _usd_digest() -> str | None:
    """Aggregate sha256 (16 hex) over the Newton-consumed USD layers.

    The generated asset is gitignored, so runtime proofs must pin its content
    digest rather than rely on the source-tree freeze hash.
    """
    import hashlib

    from isaaclab_tasks.contrib.sysid.config.franka_fr3.fr3_sysid_env_cfg import FR3_USD_PATH

    root = Path(FR3_USD_PATH).parent
    files = sorted(root.rglob("*.usda"))
    if not files:
        return None
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()[:16]


def _resolve_timing(ds, shaper) -> tuple[float, float, int]:
    """Cross-check the command rate and derive (cmd_rate, physics_rate, substeps).

    The command-rate hint (when stamped) is the rate of the command path —
    i.e. 1/dt. It is NOT the physics/shaper rate; those are decoupled: the
    driver's internal shaper + joint impedance run at physics_rate (1 kHz on
    the FR3) regardless of the command rate, and the sim steps at the same
    rate both for shaper fidelity and to resolve the stiff wrist joints.
    """
    if args_cli.controller_update_rate is not None:
        cmd_rate = float(args_cli.controller_update_rate)
    elif "controller_update_rate_hint" in ds.raw:
        cmd_rate = float(np.asarray(ds.raw["controller_update_rate_hint"]).reshape(-1)[0])
    else:
        cmd_rate = 1.0 / ds.dt
    if abs(cmd_rate * ds.dt - 1.0) > 0.01:
        raise ValueError(
            f"command-rate hint {cmd_rate:.1f} Hz disagrees with the data time axis "
            f"({1.0 / ds.dt:.1f} Hz) — refusing to guess which one is wrong."
        )

    physics_rate = shaper.rate_hz if shaper.rate_hz is not None else float(args_cli.physics_rate)
    substeps_f = ds.dt * physics_rate
    substeps = round(substeps_f)
    if abs(substeps_f - substeps) > 1e-6 or substeps < 1:
        raise ValueError(
            f"physics rate {physics_rate} Hz is not an integer multiple of the data rate "
            f"{1.0 / ds.dt:.3f} Hz (substeps {substeps_f}) — cannot model the hold exactly."
        )
    return cmd_rate, physics_rate, substeps


def _reconstruct_targets(ds, shaper, substeps: int, physics_rate: float) -> tuple[np.ndarray, float]:
    """Returns (shaped targets, loss burn-in seconds).

    shaped[i, s, :] = drive target during physics tick s of command sample i.
    Columns stay in DATA order — the shaper permutes its per-joint limits to
    the dataset's column order (validated against the FR3 canonical set).
    Without a shaper the raw target is zero-order-held by env decimation.

    The burn-in mask is a conservative diagnostic HEURISTIC: the stream is
    reconstructed from an envelope of plausible initial states (settled at
    des[0], at the measured pose, and at ± offset perturbations) and the mask
    covers every tick where any alternate diverges > 0.1 mrad, times a 1.5x
    plant-settling factor. It is NOT a bound on unknown EMA/Ruckig internal
    velocity state or candidate-dependent plant memory — real-data acceptance
    stays gated on driver-exported applied targets.
    """
    if args_cli.loss_burn_in_s is not None:
        burn_override = require_finite("loss_burn_in_s override", args_cli.loss_burn_in_s, lo=0.0)
    else:
        burn_override = None

    des_np = ds.des_dof_pos.numpy().astype(float)
    if shaper.type != "franka_fr3":
        return des_np[:, None, :], (burn_override if burn_override is not None else 0.0)

    from fr3_target_shaping import shape_targets

    print(
        f"[INFO]: reconstructing franka_fr3 clamp+EMA+Ruckig targets (APPROXIMATE, "
        f"relative_dynamics={shaper.relative_dynamics}, ema_alpha={shaper.ema_alpha}, "
        f"2x{des_np.shape[0]}x{substeps} ticks)..."
    )
    kwargs = dict(
        substeps=substeps,
        joint_indices=canonical_indices(ds.joint_names),
        relative_dynamics=shaper.relative_dynamics,
        ema_alpha=shaper.ema_alpha,
        ctrl_dt=1.0 / physics_rate,
    )
    # Settled post-homing-dwell shaper state (the primary model of the truth).
    shaped_np = shape_targets(des_np, init_target=des_np[0], **kwargs)
    # Envelope of plausible unobserved initial states — a conservative
    # diagnostic HEURISTIC, not a bound (does not cover EMA/Ruckig internal
    # velocity state or candidate-dependent plant memory; real fits stay
    # gated on driver-exported applied targets).
    dof0 = ds.dof_pos[0].numpy().astype(float)
    delta = np.maximum(0.1, 2.0 * np.abs(des_np[0] - dof0))
    alternates = [
        shape_targets(des_np, init_target=dof0, **kwargs),
        shape_targets(des_np, init_target=des_np[0] + delta, **kwargs),
        shape_targets(des_np, init_target=des_np[0] - delta, **kwargs),
    ]
    burn_in_s = (
        burn_override
        if burn_override is not None
        else measure_convergence_burn_in(shaped_np, alternates, tick_dt=1.0 / physics_rate)
    )
    deviation = float(np.abs(shaped_np[:, -1, :] - des_np).max())
    print(f"[INFO]: max |shaped - raw| target deviation: {deviation:.4f} rad; burn-in {burn_in_s:.3f} s (measured).")
    return shaped_np, burn_in_s


def _resolve_fit_joints(ds) -> list[str]:
    """Priority: CLI override > data's active_joint_names > full robot."""
    if args_cli.joint_order:
        from data_contract import normalize_joint_name

        joint_order = [normalize_joint_name(n) for n in args_cli.joint_order]
        print(f"[INFO]: Using --joint_order override: {joint_order}")
    elif ds.active_joint_names:
        joint_order = list(ds.active_joint_names)
        print(f"[INFO]: Inferred joint_order from active_joint_names: {joint_order}")
    else:
        joint_order = list(ds.joint_names)
        print(f"[INFO]: No active_joint_names in data — fitting all {len(joint_order)} joints.")
    return joint_order


def _load_params_artifact(path: str, joint_order: list[str], device) -> torch.Tensor:
    """Load a (2N,) parameter artifact; verify stored joint_order when present."""
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict):
        stored_order = blob.get("joint_order")
        if stored_order is not None and list(stored_order) != list(joint_order):
            raise ValueError(
                f"artifact {path} was fitted on joint_order {list(stored_order)} but this run "
                f"uses {list(joint_order)} — refusing to apply parameters across orders."
            )
        params = blob["sim_params"].to(device)
    else:
        print(f"[WARN]: {path} carries no joint_order metadata — assuming it matches {list(joint_order)}.")
        params = blob.to(device)
    params = params.reshape(-1)
    if params.shape[0] != 2 * len(joint_order):
        raise ValueError(f"artifact {path} shape {tuple(params.shape)} != (2*{len(joint_order)},)")
    return params


def _resolve_initial_mean(ds, joint_order: list[str], col_indices: list[int], device) -> torch.Tensor | None:
    """CMA-ES warm-start mean: explicit file > dataset kp_used/kd_used > None."""
    if args_cli.initial_mean_path:
        print(f"[INFO]: Warm-starting CMA-ES from {args_cli.initial_mean_path}")
        return _load_params_artifact(args_cli.initial_mean_path, joint_order, device)
    if args_cli.warmstart_from_data:
        print("[INFO]: Warm-starting CMA-ES from dataset kp_used/kd_used.")
        return torch.cat([ds.kp_used.to(device)[col_indices], ds.kd_used.to(device)[col_indices]])
    return None


def _load_validated_dataset():
    ds = validate_contract(
        load_dataset(args_cli.data),
        allow_stale_fraction=args_cli.allow_stale_fraction,
        allow_missing_freshness=args_cli.allow_missing_freshness,
        allow_truncated=args_cli.allow_truncated,
    )
    if ds.stale_fraction > 0:
        print(f"[WARN]: {ds.stale_fraction:.1%} stale rows accepted via --allow_stale_fraction — masked from the loss.")
    if args_cli.allow_truncated:
        print("[WARN]: --allow_truncated set — truncated/aborted runs accepted; results are DIAGNOSTIC ONLY.")
    return ds


def main() -> None:
    # ------------------------------------------------------------------ dataset (before env: it sets timing)
    ds = _load_validated_dataset()
    time_steps = ds.time.shape[0]
    dt_data = ds.dt

    shaper = resolve_shaper(
        ds.raw,
        cli_shaping=args_cli.shaping,
        cli_ema_alpha=args_cli.ema_alpha,
        cli_relative_dynamics=args_cli.relative_dynamics,
        cli_rate_hz=args_cli.shaper_rate_hz,
    )
    cmd_rate, physics_rate, substeps = _resolve_timing(ds, shaper)
    shaped_np, burn_in_s = _reconstruct_targets(ds, shaper, substeps, physics_rate)

    # The combined loss mask (burn-in + stale rows) — raises if nothing survives.
    loss_mask = build_loss_mask(time_steps, dt_data, burn_in_s, ds.state_fresh)
    scored_samples = int(loss_mask.sum())
    if scored_samples < time_steps // 2:
        print(f"[WARN]: loss mask covers {time_steps - scored_samples}/{time_steps} samples (> 50%).")

    # ------------------------------------------------------------------ env cfg
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.dt = 1.0 / physics_rate
    # With shaping the target changes every physics tick -> env steps at the
    # physics rate; without it the hold is delegated to env decimation.
    env_cfg.decimation = 1 if shaped_np.shape[1] > 1 else substeps
    env_cfg.sim.render_interval = env_cfg.decimation
    # Never let a time_out reset fire mid-replay.
    env_cfg.episode_length_s = (time_steps + 10) * dt_data
    # Effective graph mode (resolve_task_config may return either the PresetCfg
    # or the already-resolved NewtonCfg) — recorded into the run provenance.
    _phys = getattr(env_cfg.sim.physics, "newton_mjwarp", env_cfg.sim.physics)
    use_cuda_graph = getattr(_phys, "use_cuda_graph", None)
    print(
        f"[INFO]: timing: data {1.0 / dt_data:.1f} Hz (cmd path {cmd_rate:.0f} Hz), physics "
        f"{physics_rate:.0f} Hz -> sim.dt={env_cfg.sim.dt:.4f}s, decimation={env_cfg.decimation}, "
        f"targets/sample={shaped_np.shape[1]}, shaping={shaper.type}, burn-in={burn_in_s:.3f}s, "
        f"use_cuda_graph={use_cuda_graph}."
    )

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        device = env.unwrapped.device
        articulation = env.unwrapped.scene["robot"]
        num_envs = env.unwrapped.num_envs

        # The replay indexes the action vector by articulation joint order —
        # valid only if the action term covers every joint in that order.
        assert env.unwrapped.action_manager.total_action_dim == articulation.num_joints, (
            "sysid action term must cover all joints (joint_names=['.*'])"
        )

        # -------------------------------------------------------------- joint bookkeeping
        target_dof_pos_full = ds.des_dof_pos.to(device)
        measured_dof_pos_full = ds.dof_pos.to(device)
        N_data = target_dof_pos_full.shape[1]
        data_joint_order = ds.joint_names

        joint_order = _resolve_fit_joints(ds)

        sim_joint_ids = torch.tensor([articulation.joint_names.index(n) for n in joint_order], device=device)
        sim_full_joint_ids = torch.tensor([articulation.joint_names.index(n) for n in data_joint_order], device=device)
        col_indices = [data_joint_order.index(j) for j in joint_order]
        K = len(joint_order)

        measured_dof_pos = measured_dof_pos_full[:, col_indices]  # [T, N_active]
        initial_dof_pos = measured_dof_pos[0].unsqueeze(0).expand(num_envs, -1)
        # ALL joints start from the MEASURED state — the real arm never sits
        # exactly on the command.
        initial_dof_pos_full = measured_dof_pos_full[0].unsqueeze(0).expand(num_envs, -1)
        initial_dof_vel_full = ds.dof_vel.to(device)[0].unsqueeze(0).expand(num_envs, -1)

        active_col_set = set(col_indices)
        inactive_col_indices = [i for i in range(N_data) if i not in active_col_set]
        sim_inactive_joint_ids = sim_full_joint_ids[inactive_col_indices] if inactive_col_indices else None

        # Asset-default gains, captured before any candidate write (baselines).
        default_stiffness = articulation.data.joint_stiffness.torch[0, sim_joint_ids].detach().clone()
        default_damping = articulation.data.joint_damping.torch[0, sim_joint_ids].detach().clone()

        # kp_used/kd_used are provenance (never the fit target). When fitting a
        # subset, the held joints still shape the coupled dynamics: give them
        # the recorded rig gains instead of the asset placeholders.
        if sim_inactive_joint_ids is not None:
            kp_used_dev = ds.kp_used.to(device)
            kd_used_dev = ds.kd_used.to(device)
            articulation.write_joint_stiffness_to_sim_index(
                stiffness=kp_used_dev[inactive_col_indices].unsqueeze(0).expand(num_envs, -1),
                joint_ids=sim_inactive_joint_ids,
            )
            articulation.write_joint_damping_to_sim_index(
                damping=kd_used_dev[inactive_col_indices].unsqueeze(0).expand(num_envs, -1),
                joint_ids=sim_inactive_joint_ids,
            )
            print(f"[INFO]: held joints {inactive_col_indices} use dataset kp_used/kd_used gains.")

        print(f"[INFO]: Trajectory: {time_steps} steps, {float(ds.time[-1]):.1f}s")

        # -------------------------------------------------------------- replay machinery
        shaped_targets = torch.as_tensor(shaped_np, dtype=torch.float32, device=device)
        steps_per_sample = shaped_targets.shape[1]
        default_joint_pos = articulation.data.default_joint_pos.torch
        actions = torch.zeros(num_envs, articulation.num_joints, device=device)

        applied_torque = getattr(articulation.data, "applied_torque", None)
        effort_limits = getattr(articulation.data, "joint_effort_limits", None)
        sat_available = applied_torque is not None and effort_limits is not None
        if sat_available:
            effort_limits_t = effort_limits.torch[0, sim_joint_ids].detach().clone()
        else:
            print("[WARN]: applied_torque/joint_effort_limits unavailable — saturation metric disabled.")

        def apply_params(params: torch.Tensor) -> None:
            """Write per-env {stiffness, damping} for the fitted joints + the initial state."""
            stiffness, damping = params[:, :K], params[:, K:]
            articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness, joint_ids=sim_joint_ids)
            articulation.write_joint_damping_to_sim_index(damping=damping, joint_ids=sim_joint_ids)
            # Telemetry mirror (a single all-joint group may alias the sim
            # array — the write is then redundant but harmless).
            for actuator in articulation.actuators.values():
                for local_idx, joint_name in enumerate(actuator.joint_names):
                    if joint_name in joint_order:
                        k = joint_order.index(joint_name)
                        actuator.stiffness[:, local_idx] = stiffness[:, k]
                        actuator.damping[:, local_idx] = damping[:, k]
            articulation.write_joint_position_to_sim_index(position=initial_dof_pos, joint_ids=sim_joint_ids)
            articulation.write_joint_velocity_to_sim_index(
                velocity=initial_dof_vel_full[:, col_indices], joint_ids=sim_joint_ids
            )
            if sim_inactive_joint_ids is not None:
                articulation.write_joint_position_to_sim_index(
                    position=initial_dof_pos_full[:, inactive_col_indices], joint_ids=sim_inactive_joint_ids
                )
                articulation.write_joint_velocity_to_sim_index(
                    velocity=initial_dof_vel_full[:, inactive_col_indices], joint_ids=sim_inactive_joint_ids
                )

        def replay_once(params: torch.Tensor, opt: CMAESOptimizer | None = None):
            """Roll the full trajectory with one parameter population.

            Returns (scores, saturation_fraction) per env: burn-in-masked mean
            over counted steps of the sum-over-joints squared error, and the
            fraction of (tick, joint) samples at >=99% of the effort limit.
            When ``opt`` is given, samples also feed its trajectory buffer.
            """
            env.reset()
            apply_params(params)
            scores = torch.zeros(num_envs, device=device)
            sat_events = torch.zeros(num_envs, device=device)
            counted, ticks = 0, 0
            with torch.inference_mode():
                for i in range(time_steps):
                    count = bool(loss_mask[i])  # burn-in + stale rows (data_contract.build_loss_mask)
                    sim_q = articulation.data.joint_pos.torch[:, sim_joint_ids]
                    real_q = measured_dof_pos[i].unsqueeze(0).expand(num_envs, -1)
                    if opt is not None:
                        opt.tell(sim_q, real_q, count=count)
                    if count:
                        scores += torch.sum(torch.square(sim_q - real_q), dim=1)
                        counted += 1
                    # Action index order == articulation joint order ('.*').
                    for s in range(steps_per_sample):
                        actions[:, sim_full_joint_ids] = shaped_targets[i, s] - default_joint_pos[:, sim_full_joint_ids]
                        _, _, terminated, truncated, _ = env.step(actions)
                        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                            raise RuntimeError(f"env terminated/truncated mid-replay at sample {i}")
                        if sat_available:
                            tau = applied_torque.torch[:, sim_joint_ids].abs()
                            sat_events += (tau >= 0.99 * effort_limits_t).float().mean(dim=1)
                            ticks += 1
            return scores / max(counted, 1), sat_events / max(ticks, 1)

        # -------------------------------------------------------------- eval-only mode
        if args_cli.eval_params:
            if num_envs < 3:
                raise ValueError(
                    "eval mode requires --num_envs >= 3 (candidate + recorded-gain + asset-default "
                    "baselines); a vacuous single-row eval must not emit a PASS."
                )
            eval_params = _load_params_artifact(args_cli.eval_params, joint_order, device)
            rows = eval_params.unsqueeze(0).repeat(num_envs, 1)
            labels = {0: "eval_params"}
            if num_envs >= 2:
                rows[1] = torch.cat([ds.kp_used.to(device)[col_indices], ds.kd_used.to(device)[col_indices]])
                labels[1] = "recorded_gains"
            if num_envs >= 3:
                rows[2] = torch.cat([default_stiffness, default_damping])
                labels[2] = "asset_default_gains"
            scores, sat = replay_once(rows)
            result = {
                "data": args_cli.data,
                "eval_params": args_cli.eval_params,
                "joint_order": joint_order,
                "burn_in_s": burn_in_s,
                "stale_fraction": ds.stale_fraction,
                "usd_digest": _usd_digest(),
                "shaping": shaper.type,
                "rows": {
                    label: {"score_rad2": scores[idx].item(), "saturation": sat[idx].item()}
                    for idx, label in labels.items()
                },
            }
            baselines = [label for label in labels.values() if label != "eval_params"]
            result["beats"] = {
                label: bool(scores[0].item() < result["rows"][label]["score_rad2"]) for label in baselines
            }
            # Verdict policy: beat every baseline AND stay under the saturation ceiling.
            result["saturation_ceiling"] = args_cli.eval_max_saturation
            result["saturation_ok"] = bool(sat[0].item() <= args_cli.eval_max_saturation)
            result["pass"] = all(result["beats"].values()) and result["saturation_ok"]
            print(f"\n[EVAL] dataset: {args_cli.data} (burn-in {burn_in_s:.3f}s, shaping {shaper.type})")
            for idx, label in labels.items():
                print(f"[EVAL] {label:<24} score {scores[idx].item():.6e} rad²  sat {sat[idx].item():.3%}")
            print(f"[EVAL] verdict: {'PASS' if result['pass'] else 'FAIL'} (beats: {result['beats']})")
            out_path = Path(args_cli.eval_params).with_name("eval_result.json")
            out_path.write_text(json.dumps(result, indent=2))
            print(f"[EVAL] machine-readable result → {out_path}")
            env.close()
            if not result["pass"]:
                sys.exit(1)
            return

        # -------------------------------------------------------------- optimizer
        bounds = build_bounds(joint_order).to(device)
        initial_mean = _resolve_initial_mean(ds, joint_order, col_indices, device)

        log_dir = args_cli.log_dir or str(Path("logs") / "sysid" / env_cfg.sysid.robot_name)
        os.makedirs(log_dir, exist_ok=True)

        opt = CMAESOptimizer(
            bounds=bounds,
            population_size=num_envs,
            log_dir=log_dir,
            joint_order=joint_order,
            max_iteration=args_cli.max_iterations or env_cfg.sysid.cmaes.max_iteration,
            data={"time": ds.time, "dof_pos": ds.dof_pos, "des_dof_pos": ds.des_dof_pos},
            device=device,
            epsilon=env_cfg.sysid.cmaes.epsilon,
            sigma=env_cfg.sysid.cmaes.sigma,
            save_interval=env_cfg.sysid.cmaes.save_interval,
            save_optimization_process=env_cfg.sysid.cmaes.save_optimization_process,
            initial_mean=initial_mean,
            warmstart_sigma_scale=args_cli.warmstart_sigma_scale,
            plateau_patience=env_cfg.sysid.cmaes.plateau_patience,
            plateau_min_delta=env_cfg.sysid.cmaes.plateau_min_delta,
            seed=args_cli.seed,
            run_metadata={
                "data": args_cli.data,
                "task": args_cli.task,
                "seed": args_cli.seed,
                "shaping": shaper.type,
                "shaper_ema_alpha": shaper.ema_alpha,
                "shaper_relative_dynamics": shaper.relative_dynamics,
                "shaper_rate_hz": shaper.rate_hz,
                "shaper_approximate": shaper.approximate,
                "burn_in_s": burn_in_s,
                "stale_fraction": ds.stale_fraction,
                "allow_stale_fraction": args_cli.allow_stale_fraction,
                "allow_missing_freshness": args_cli.allow_missing_freshness,
                "allow_truncated": args_cli.allow_truncated,
                "usd_digest": _usd_digest(),
                "scored_samples": scored_samples,
                "cmd_rate_hz": cmd_rate,
                "physics_rate_hz": physics_rate,
                "use_cuda_graph": use_cuda_graph,
                "zero_gravity": env_cfg.sysid.zero_gravity,
            },
        )

        # -------------------------------------------------------------- fit loop
        last_sat = torch.zeros(num_envs, device=device)
        while True:
            _, last_sat = replay_once(opt.sim_params, opt=opt)
            opt.writer.add_scalar("0_Episode/saturation_mean", last_sat.mean().item(), opt.iteration_counter)
            opt.writer.add_scalar("0_Episode/saturation_max", last_sat.max().item(), opt.iteration_counter)
            opt.evolve()
            if opt.finished():
                break
            print(f"[INFO]: generation {opt.iteration_counter} done (sat mean {last_sat.mean().item():.3%}).")

        # Reroll the CMA mean so the reported mean is an EVALUATED result.
        mean_params = opt.get_best_sim_params()
        mean_scores, mean_sat = replay_once(mean_params.unsqueeze(0).expand(num_envs, -1))
        mean_score = mean_scores[0].item()
        best_params_ever, best_score_ever, best_iter_ever = opt.get_best_ever()

        # -------------------------------------------------------------- summary
        fit_log_dir = Path(opt.writer.log_dir)
        final_params = mean_params.detach().cpu()
        final_scores = opt.scores_buffer[opt.iteration_counter - 1, :].detach().cpu()
        summary_path = fit_log_dir / "fitted_parameters.txt"
        header_w = max(len(j) for j in joint_order) + 2

        def write_table(f, params: torch.Tensor) -> None:
            f.write(f"{'joint':<{header_w}}{'stiffness [Nm/rad]':>22}{'damping [Nm·s/rad]':>22}\n")
            f.write("-" * (header_w + 44) + "\n")
            for i, jname in enumerate(joint_order):
                f.write(f"{jname:<{header_w}}{params[i].item():>22.4f}{params[K + i].item():>22.4f}\n")

        with open(summary_path, "w") as f:
            f.write("# CMA-ES sysid fitted implicit-actuator gains\n")
            f.write(f"# run dir : {fit_log_dir}\n")
            f.write(f"# data    : {args_cli.data}\n")
            f.write(f"# task    : {args_cli.task}\n")
            f.write(
                f"# timing  : data {1.0 / dt_data:.1f} Hz (cmd {cmd_rate:.0f} Hz), physics"
                f" {physics_rate:.0f} Hz, targets/sample {steps_per_sample}\n"
            )
            shaping_label = shaper.type
            if shaper.type != "none":
                shaping_label += f" (relative_dynamics={shaper.relative_dynamics}, ema_alpha={shaper.ema_alpha})" + (
                    " APPROXIMATE reconstruction" if shaper.approximate else ""
                )
            f.write(f"# shaping : {shaping_label}\n")
            f.write(f"# burn-in : {burn_in_s:.3f} s loss mask (shaper warm-up)\n")
            if args_cli.allow_truncated:
                f.write("# WARNING : TRUNCATED/ABORTED RUN accepted via --allow_truncated — DIAGNOSTIC ONLY\n")
            f.write(
                f"# samples : {scored_samples}/{time_steps} scored"
                + (
                    f" — {ds.stale_fraction:.1%} stale rows masked via --allow_stale_fraction (debug-only)\n"
                    if ds.stale_fraction > 0
                    else "\n"
                )
            )
            f.write(f"# gravity : {'disabled (compensated plant)' if env_cfg.sysid.zero_gravity else 'enabled'}\n")
            f.write(
                "# NOTE    : gains are EFFECTIVE replay gains for this command path and excitation band —\n"
                "#           they absorb friction, armature, payload, compensation, filtering and delay.\n"
            )
            f.write(f"# final generation : {opt.iteration_counter}\n\n")
            f.write(f"# CMA mean — rerolled, evaluated score {mean_score:.6e} rad², sat {mean_sat[0].item():.3%}\n")
            write_table(f, final_params)
            if best_params_ever is not None:
                f.write(f"\n# Best evaluated candidate (score {best_score_ever:.6e}, generation {best_iter_ever})\n")
                write_table(f, best_params_ever.detach().cpu())
            f.write(
                f"\nfinal population score [rad²]:  min={final_scores.min().item():.6e}  "
                f"max={final_scores.max().item():.6e}  mean={final_scores.mean().item():.6e}\n"
            )
            f.write(
                f"final population saturation:    mean={last_sat.mean().item():.3%}  max={last_sat.max().item():.3%}\n"
            )
        print(f"[INFO]: Wrote fitted parameter summary → {summary_path}")

        # Machine-readable fit result (mirrors the summary header + scores).
        fit_result = {
            **opt.run_metadata,
            "joint_order": joint_order,
            "final_generation": opt.iteration_counter,
            "mean_rerolled_score_rad2": mean_score,
            "mean_saturation": mean_sat[0].item(),
            "best_evaluated_score_rad2": best_score_ever,
            "best_evaluated_generation": best_iter_ever,
            "final_population_score_min": final_scores.min().item(),
            "final_population_score_max": final_scores.max().item(),
            "final_population_saturation_mean": last_sat.mean().item(),
        }
        (fit_log_dir / "fit_result.json").write_text(json.dumps(fit_result, indent=2))

        opt.close()
        env.close()

    # ------------------------------------------------------------------ plot (outside sim context)
    _run_plot(fit_log_dir)


def _run_plot(fit_log_dir: Path) -> None:
    """Overlay plot via isaac_ros_sysid's plot_chirp.py, when available."""
    if args_cli.plot_script:
        plot_script = Path(args_cli.plot_script)
        if plot_script.exists():
            import subprocess

            plot_out = fit_log_dir / "fit_signals.png"
            env_vars = dict(os.environ)
            env_vars["PYTHONPATH"] = f"{plot_script.parents[1]}:{env_vars.get('PYTHONPATH', '')}"
            print(f"[INFO]: Plotting fit signals → {plot_out}")
            result = subprocess.run(
                [
                    sys.executable,
                    str(plot_script),
                    "--data",
                    args_cli.data,
                    "--fit",
                    str(fit_log_dir),
                    "--out",
                    str(plot_out),
                ],
                env=env_vars,
            )
            if result.returncode != 0:
                print(f"[WARN]: plot_chirp.py exited with code {result.returncode}")
        else:
            print(f"[WARN]: plot script not found at {plot_script}; skipping.")


if __name__ == "__main__":
    main()
