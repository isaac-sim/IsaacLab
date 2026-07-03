# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset loading, contract validation, and provenance resolution for sysid fitting.

Pure numpy/torch-CPU — no isaaclab imports — so every rule here is unit-testable
(see ``tests/test_data_contract.py``). The contract is fail-closed: anything a
fit would silently mis-consume (permuted columns, corrupted time base, missing
or non-finite gains, missing shaper provenance, clamped commands, malformed
safety metadata) raises :class:`ContractError` instead.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import torch

CANONICAL_JOINT_ORDER: list[str] = [f"fr3_joint{i}" for i in range(1, 8)]

# Fields a fit consumes; all fail-closed per the ratified metadata contract.
REQUIRED_KEYS = (
    "time",
    "des_dof_pos",
    "dof_pos",
    "dof_vel",
    "dof_tau_est",
    "joint_names",
    "active_joint_names",
    "sample_rate",
    "kp_used",
    "kd_used",
)
SHAPER_PARAM_KEYS = ("shaper_ema_alpha", "shaper_relative_dynamics", "shaper_rate_hz")

_PANDA_RE = re.compile(r"^panda_joint(\d+)$")


class ContractError(ValueError):
    """A dataset violates the sysid data contract."""


def normalize_joint_name(name: str) -> str:
    """robot-control / franka drivers report panda_joint* even on the FR3."""
    m = _PANDA_RE.match(name)
    return f"fr3_joint{m.group(1)}" if m else name


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def require_finite(name: str, value, lo: float | None = None, hi: float | None = None, lo_open: bool = False) -> float:
    """Validate a scalar: finite, and inside [lo, hi] (lo exclusive when lo_open)."""
    try:
        v = _to_float(value)
    except (TypeError, ValueError) as e:
        raise ContractError(f"{name} is not a number: {value!r}") from e
    if not math.isfinite(v):
        raise ContractError(f"{name} is not finite: {value!r}")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ContractError(f"{name}={v} out of range (must be {'>' if lo_open else '>='} {lo})")
    if hi is not None and v > hi:
        raise ContractError(f"{name}={v} out of range (must be <= {hi})")
    return v


def _to_str(value) -> str:
    if isinstance(value, np.ndarray):
        return str(value.reshape(-1)[0])
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _to_tensor64(value) -> torch.Tensor:
    """float64 conversion — epoch-scale stamps collapse at float32 precision
    (1.75e9 s + 5 ms increments are indistinguishable in fp32)."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double()
    return torch.as_tensor(np.asarray(value), dtype=torch.float64)


def _to_name_list(value) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    return [str(v) for v in value]


def load_dataset(path: str) -> dict:
    """Load a chirp dataset (.pt or .npz) into a plain dict (CPU)."""
    if path.endswith(".npz"):
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class Dataset:
    """Validated dataset with normalized joint names and float32 CPU tensors."""

    time: torch.Tensor  # (T,)
    des_dof_pos: torch.Tensor  # (T, N)
    dof_pos: torch.Tensor  # (T, N)
    dof_vel: torch.Tensor  # (T, N)
    dof_tau_est: torch.Tensor  # (T, N) — diagnostics/saturation only, never in the loss
    joint_names: list[str]  # normalized, length N
    active_joint_names: list[str]  # normalized subset (the fitted joints)
    kp_used: torch.Tensor  # (N,)
    kd_used: torch.Tensor  # (N,)
    sample_rate: float
    dt: float  # nominal 1/sample_rate
    # Per-row state freshness (1 = the collector read a NEW JointState for this
    # row; 0 = it reused a stale one and dof_pos/dof_vel repeat). None when the
    # dataset predates the key.
    state_fresh: torch.Tensor | None
    stale_fraction: float
    raw: dict  # original mapping for provenance keys


def _validate_shapes(data: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    time = _to_tensor(data["time"]).reshape(-1)
    des = _to_tensor(data["des_dof_pos"])
    pos = _to_tensor(data["dof_pos"])
    vel = _to_tensor(data["dof_vel"])
    tau = _to_tensor(data["dof_tau_est"])
    if des.ndim != 2:
        raise ContractError(f"des_dof_pos must be 2-D, got {tuple(des.shape)}")
    T, N = des.shape
    for key, tensor, shape in (("dof_pos", pos, (T, N)), ("dof_vel", vel, (T, N)), ("dof_tau_est", tau, (T, N))):
        if tensor.shape != shape:
            raise ContractError(f"{key} shape {tuple(tensor.shape)} != {shape}")
    if time.shape[0] != T:
        raise ContractError(f"time has {time.shape[0]} entries for {T} rows")
    if T < 2:
        raise ContractError(f"trajectory too short: T={T}")
    checks = (("time", time), ("des_dof_pos", des), ("dof_pos", pos), ("dof_vel", vel), ("dof_tau_est", tau))
    for key, tensor in checks:
        if not torch.isfinite(tensor).all():
            raise ContractError(f"{key} contains non-finite values")
    return time, des, pos, vel, tau


def _validate_gains(data: dict, N: int) -> tuple[torch.Tensor, torch.Tensor]:
    kp = _to_tensor(data["kp_used"]).reshape(-1)
    kd = _to_tensor(data["kd_used"]).reshape(-1)
    if kp.shape[0] != N or kd.shape[0] != N:
        raise ContractError(f"kp_used/kd_used length {kp.shape[0]}/{kd.shape[0]} != {N}")
    if not torch.isfinite(kp).all() or not torch.isfinite(kd).all():
        raise ContractError("kp_used/kd_used contain non-finite values")
    if (kp <= 0).any() or (kd < 0).any():
        raise ContractError(f"kp_used must be > 0 and kd_used >= 0: kp={kp.tolist()}, kd={kd.tolist()}")
    return kp, kd


def _validate_freshness(
    data: dict, T: int, allow_stale_fraction: float, allow_missing_freshness: bool
) -> tuple[torch.Tensor | None, float]:
    """Stale-state gate: rows that reused an old JointState bias the fit.

    Both ``state_fresh`` and ``state_stamps`` are required (absence would let a
    stripped dataset evade the gate) unless ``allow_missing_freshness`` is set
    explicitly. The mask must be self-consistent with the stamps: row 0 fresh,
    and fresh[i] == (stamps[i] > stamps[i-1]). Fail-closed by default
    (stale_fraction must be 0); ``allow_stale_fraction`` is the explicit
    debug-only override — the fitter then MASKS stale rows from the loss,
    which is still NOT timestamp-aware alignment (q is observed at the stamp
    time but compared to the sim at command-grid time).
    """
    has_fresh, has_stamps = "state_fresh" in data, "state_stamps" in data
    if not (has_fresh and has_stamps):
        if allow_missing_freshness:
            return None, 0.0
        raise ContractError(
            f"dataset lacks {'state_fresh ' if not has_fresh else ''}{'state_stamps' if not has_stamps else ''}"
            " — collected datasets must stamp both (a stripped dataset would evade the stale-row "
            "gate). Pass --allow_missing_freshness only for legacy data you trust."
        )

    state_fresh = _to_tensor(data["state_fresh"]).reshape(-1)
    stamps = _to_tensor64(data["state_stamps"]).reshape(-1)
    if state_fresh.shape[0] != T or stamps.shape[0] != T:
        raise ContractError(
            f"state_fresh/state_stamps have {state_fresh.shape[0]}/{stamps.shape[0]} entries for {T} rows"
        )
    if not ((state_fresh == 0) | (state_fresh == 1)).all():
        raise ContractError("state_fresh must be 0/1 valued")
    if not torch.isfinite(stamps).all():
        raise ContractError("state_stamps contain non-finite values")
    if (stamps[1:] - stamps[:-1] < 0).any():
        raise ContractError("state_stamps are not monotonically non-decreasing")
    if state_fresh[0] != 1:
        raise ContractError("state_fresh[0] must be 1 (the first row cannot reuse a prior state)")
    expected_fresh = (stamps[1:] > stamps[:-1]).float()  # fp64 comparison — see _to_tensor64
    if not torch.equal(state_fresh[1:], expected_fresh):
        bad = int((state_fresh[1:] != expected_fresh).sum())
        raise ContractError(
            f"state_fresh is inconsistent with state_stamps on {bad} rows (fresh[i] must equal stamps[i] > stamps[i-1])"
        )

    stale_fraction = float(1.0 - state_fresh.mean())
    if stale_fraction > allow_stale_fraction:
        raise ContractError(
            f"{stale_fraction:.1%} of rows reused a stale JointState (state_fresh=0) — "
            f"scoring them against a fresh sim trajectory biases the fit. Rejected "
            f"(allow_stale_fraction={allow_stale_fraction:.1%}); pass --allow_stale_fraction "
            "to mask stale rows from the loss instead (debug-only, not acceptance-grade)."
        )
    return state_fresh, stale_fraction


# PM-ruled hard ceiling on the stale-row debug override.
MAX_ALLOW_STALE_FRACTION = 0.20


def validate_contract(
    data: dict,
    allow_stale_fraction: float = 0.0,
    allow_missing_freshness: bool = False,
    allow_truncated: bool = False,
) -> Dataset:
    """Fail-closed structural validation. Raises :class:`ContractError`."""
    allow_stale_fraction = require_finite(
        "allow_stale_fraction", allow_stale_fraction, lo=0.0, hi=MAX_ALLOW_STALE_FRACTION
    )
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ContractError(f"dataset is missing required keys: {missing}")

    time, des, pos, vel, tau = _validate_shapes(data)
    T, N = des.shape

    joint_names = [normalize_joint_name(n) for n in _to_name_list(data["joint_names"])]
    if len(joint_names) != N:
        raise ContractError(f"joint_names has {len(joint_names)} entries for {N} columns")
    if len(set(joint_names)) != N:
        raise ContractError(f"duplicate joint names after normalization: {joint_names}")

    active = [normalize_joint_name(n) for n in _to_name_list(data["active_joint_names"])]
    if not active:
        raise ContractError("active_joint_names is empty")
    if len(set(active)) != len(active):
        raise ContractError(f"duplicate active joint names after normalization: {active}")
    unknown = [n for n in active if n not in joint_names]
    if unknown:
        raise ContractError(f"active_joint_names not present in joint_names: {unknown}")

    sample_rate = require_finite("sample_rate", data["sample_rate"], lo=0, lo_open=True)
    dt = 1.0 / sample_rate
    diffs = time[1:] - time[:-1]
    if (diffs <= 0).any():
        raise ContractError("time axis is not strictly increasing")
    max_dev = float((diffs - dt).abs().max())
    if max_dev > 0.01 * dt:
        raise ContractError(
            f"time axis is not uniform at sample_rate={sample_rate:.1f} Hz "
            f"(max |dt - {dt * 1e3:.3f} ms| = {max_dev * 1e3:.3f} ms > 1%)"
        )

    kp_used, kd_used = _validate_gains(data, N)
    state_fresh, stale_fraction = _validate_freshness(data, T, allow_stale_fraction, allow_missing_freshness)
    _validate_safety_and_completion(
        data,
        duration_s=(T - 1) * dt,
        dt=dt,
        allow_truncated=allow_truncated,
        collected=state_fresh is not None and str(data.get("mode", "")) != "synthetic",
    )

    return Dataset(
        time=time,
        des_dof_pos=des,
        dof_pos=pos,
        dof_vel=vel,
        dof_tau_est=tau,
        joint_names=joint_names,
        active_joint_names=active,
        kp_used=kp_used,
        kd_used=kd_used,
        sample_rate=sample_rate,
        dt=dt,
        state_fresh=state_fresh,
        stale_fraction=stale_fraction,
        raw=data,
    )


@dataclass
class ShaperSpec:
    """Resolved command-shaping provenance."""

    type: str  # 'franka_fr3' | 'none'
    ema_alpha: float | None = None
    relative_dynamics: float | None = None
    rate_hz: float | None = None
    # True when the drive targets are RECONSTRUCTED (not driver-exported).
    # Stays True for every franka_fr3 fit until des_dof_pos_applied is loaded,
    # schema-validated AND consumed by the replay.
    approximate: bool = False


def _gains_provenance_shaping(data: dict) -> str | None:
    """Fallback: the landed collector nests command_shaping in gains_provenance."""
    gp = data.get("gains_provenance")
    if gp is None:
        return None
    if isinstance(gp, np.ndarray):
        gp = _to_str(gp)
    if isinstance(gp, str | bytes):
        try:
            gp = json.loads(gp)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ContractError(f"gains_provenance is not valid JSON: {e}") from e
    if not isinstance(gp, dict):
        raise ContractError(f"gains_provenance must be a JSON object/dict, got {type(gp).__name__}")
    if "command_shaping" in gp:
        return str(gp["command_shaping"])
    return None


def _validate_safety_and_completion(
    data: dict, duration_s: float, dt: float, allow_truncated: bool, collected: bool
) -> None:
    """Safety/completion gates: clamped, aborted, and truncated runs.

    Clamped commands are rejected UNCONDITIONALLY — des_dof_pos is not the
    applied target, and the des_dof_pos_applied path is not an escape hatch
    until that signal is loaded, schema-validated and consumed by the replay.
    Aborted/truncated runs are rejected unless ``allow_truncated`` is set
    explicitly (debug-only, provenance-stamped). Malformed safety metadata is
    rejected, never treated as absent.
    """
    # Stripping the completion metadata must not fail open: freshness-stamped
    # (i.e. collector-produced) datasets must carry both completion keys.
    if collected and not allow_truncated:
        missing_completion = [k for k in ("safety_controller", "intended_duration_s") if k not in data]
        if missing_completion:
            raise ContractError(
                f"collector-produced dataset lacks completion metadata {missing_completion} — "
                "a stripped dataset would evade the truncation gate. Pass --allow_truncated "
                "only for diagnostics."
            )
    sc = data.get("safety_controller")
    if sc is not None:
        if isinstance(sc, np.ndarray):
            sc = _to_str(sc)
        if isinstance(sc, str | bytes):
            try:
                sc = json.loads(sc)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ContractError(f"safety_controller metadata is not valid JSON: {e}") from e
        if not isinstance(sc, dict):
            raise ContractError(f"safety_controller metadata must be a dict, got {type(sc).__name__}")
        if bool(sc.get("clamped", False)):
            raise ContractError(
                "safety controller clamped commands during this run — des_dof_pos does not "
                "reflect the applied targets. Run rejected (the des_dof_pos_applied replay "
                "path is not wired yet; until then clamped datasets are always rejected)."
            )
        if bool(sc.get("aborted", False)) and not allow_truncated:
            raise ContractError(
                "run was safety-aborted (safety_controller.aborted) — the trajectory is a "
                "truncated prefix, not a completed excitation. Pass --allow_truncated only "
                "for diagnostics."
            )
        if bool(sc.get("operator_stop", False)) and not allow_truncated:
            raise ContractError(
                "run was manually stopped (safety_controller.operator_stop) — a late stop can "
                "leave the duration inside tolerance while the excitation is incomplete. "
                "Pass --allow_truncated only for diagnostics."
            )
    if "intended_duration_s" in data:
        intended = require_finite("intended_duration_s", data["intended_duration_s"], lo=0, lo_open=True)
        # One-command-sample tolerance: 19.6 s of an intended 20 s run is 80
        # missing rows, not rounding.
        if duration_s < intended - 1.5 * dt and not allow_truncated:
            raise ContractError(
                f"trajectory covers {duration_s:.3f}s of an intended {intended:.3f}s run "
                f"({round((intended - duration_s) / dt)} rows missing) — truncated. "
                "Pass --allow_truncated only for diagnostics."
            )


def resolve_shaper(
    data: dict,
    cli_shaping: str = "auto",
    cli_ema_alpha: float | None = None,
    cli_relative_dynamics: float | None = None,
    cli_rate_hz: float | None = None,
) -> ShaperSpec:
    """Resolve which command path produced the dataset. Fail-closed.

    Precedence: explicit CLI > top-level ``shaper_type`` > the collector's
    ``gains_provenance.command_shaping`` JSON. Nested provenance is validated
    whenever present and cross-checked against the top-level key — a conflict
    is a hard failure, never silently ignored. In auto mode a franka_fr3
    shaper requires EVERY parameter key (no silent defaults); forcing via CLI
    is the explicit override that permits the documented defaults. All
    parameters are range-validated regardless of source.
    """

    def _canon(t: str) -> str:
        if t in ("none", "zoh", "identity"):
            return "none"
        if t in ("franka_fr3", "franka_fr3_ema_ruckig"):
            return "franka_fr3"
        raise ContractError(f"unknown shaper_type in dataset: '{t}'")

    # Nested provenance is parsed (and its JSON validated) whenever present.
    nested_type = _gains_provenance_shaping(data)
    top_type = _to_str(data["shaper_type"]) if "shaper_type" in data else None
    if top_type is not None and nested_type is not None and _canon(top_type) != _canon(nested_type):
        raise ContractError(
            f"shaper provenance conflict: top-level shaper_type='{top_type}' vs "
            f"gains_provenance.command_shaping='{nested_type}' — the dataset is self-inconsistent."
        )

    forced = cli_shaping != "auto"
    if forced:
        raw_type = cli_shaping
    elif top_type is not None:
        raw_type = top_type
    elif nested_type is not None:
        raw_type = nested_type
    else:
        raise ContractError(
            "dataset carries no shaper provenance (top-level shaper_type or "
            "gains_provenance.command_shaping) — pass --shaping {franka_fr3,none} "
            "explicitly; fitting must not guess the command path."
        )

    if raw_type in ("none", "zoh", "identity"):
        return ShaperSpec(type="none")
    if raw_type not in ("franka_fr3", "franka_fr3_ema_ruckig"):
        raise ContractError(f"unknown shaper_type in dataset: '{raw_type}'")

    cli_values = {
        "shaper_ema_alpha": cli_ema_alpha,
        "shaper_relative_dynamics": cli_relative_dynamics,
        "shaper_rate_hz": cli_rate_hz,
    }
    if not forced:
        missing = [k for k in SHAPER_PARAM_KEYS if k not in data and cli_values[k] is None]
        if missing:
            raise ContractError(
                f"--shaping auto resolved franka_fr3 but the dataset lacks {missing} — "
                "stamp them at collection time or pass explicit overrides."
            )

    def pick(key: str, default: float) -> float:
        if cli_values[key] is not None:
            return float(cli_values[key])
        if key in data:
            return _to_float(data[key])
        return default

    ema_alpha = require_finite("shaper_ema_alpha", pick("shaper_ema_alpha", 0.02), lo=0, hi=1.0, lo_open=True)
    relative_dynamics = require_finite(
        "shaper_relative_dynamics", pick("shaper_relative_dynamics", 0.3), lo=0, hi=1.0, lo_open=True
    )
    rate_hz = require_finite("shaper_rate_hz", pick("shaper_rate_hz", 1000.0), lo=0, lo_open=True)

    return ShaperSpec(
        type="franka_fr3",
        ema_alpha=ema_alpha,
        relative_dynamics=relative_dynamics,
        rate_hz=rate_hz,
        approximate=True,
    )


def canonical_indices(joint_names: list[str]) -> list[int]:
    """Map each data column to its canonical FR3 joint index (for per-joint limits)."""
    unknown = [n for n in joint_names if n not in CANONICAL_JOINT_ORDER]
    if unknown:
        raise ContractError(f"joints outside the FR3 canonical set: {unknown}")
    return [CANONICAL_JOINT_ORDER.index(n) for n in joint_names]


def measure_convergence_burn_in(
    primary: np.ndarray,
    alternates: list[np.ndarray] | np.ndarray,
    tick_dt: float,
    target_rad: float = 1e-4,
    floor_s: float = 0.25,
    settle_factor: float = 1.5,
) -> float:
    """Burn-in from measured shaper-stream divergence — a conservative
    diagnostic HEURISTIC, not a mathematical bound.

    ``primary`` is the shaped target stream (T, S, N) seeded settled at
    ``des[0]``; ``alternates`` are streams from other plausible initial states
    (measured pose, ± offset perturbations). The mask covers every tick where
    any alternate diverges from the primary by more than ``target_rad``,
    multiplied by ``settle_factor`` to leave room for candidate-dependent
    PLANT memory outliving the target transient. Unknown EMA/Ruckig internal
    velocity/acceleration state is NOT bracketed by position-seeded streams —
    real-data acceptance stays gated on driver-exported applied targets.
    """
    if isinstance(alternates, np.ndarray):
        alternates = [alternates]
    if not alternates:
        raise ContractError("measure_convergence_burn_in requires at least one alternate stream")
    tick_dt = require_finite("tick_dt", tick_dt, lo=0, lo_open=True)
    target_rad = require_finite("target_rad", target_rad, lo=0, lo_open=True)
    floor_s = require_finite("floor_s", floor_s, lo=0.0)
    settle_factor = require_finite("settle_factor", settle_factor, lo=1.0)
    if not np.isfinite(primary).all():
        raise ContractError("shaped target stream (primary) contains non-finite values")
    last_exceed = -1
    for alt in alternates:
        if alt.shape != primary.shape:
            raise ValueError(f"shape mismatch: {primary.shape} vs {alt.shape}")
        if not np.isfinite(alt).all():
            raise ContractError("shaped target stream (alternate) contains non-finite values")
        deviation = np.abs(primary - alt).max(axis=2).reshape(-1)  # per tick
        exceed = np.nonzero(deviation > target_rad)[0]
        if exceed.size:
            last_exceed = max(last_exceed, int(exceed[-1]))
    if last_exceed < 0:
        return floor_s
    return max(floor_s, settle_factor * float((last_exceed + 1) * tick_dt))


def build_loss_mask(
    time_steps: int,
    dt: float,
    burn_in_s: float,
    state_fresh: torch.Tensor | None,
) -> torch.Tensor:
    """Per-sample loss mask combining the burn-in window and stale rows.

    A sample is scored iff its time is past the burn-in AND its measured state
    is fresh. Raises :class:`ContractError` when nothing survives — a fit or
    eval on an empty mask would divide by zero and report garbage.
    """
    times = torch.arange(time_steps, dtype=torch.float64) * dt
    mask = times >= burn_in_s
    if state_fresh is not None:
        mask &= state_fresh.bool()
    if int(mask.sum()) == 0:
        raise ContractError(
            f"loss mask (burn-in {burn_in_s:.3f} s + stale rows) leaves zero scored samples "
            f"in the {time_steps * dt:.3f} s trajectory — record a longer/cleaner run."
        )
    return mask


def compute_burn_in_s(
    shaper: ShaperSpec,
    dof_pos0: torch.Tensor,
    des0: torch.Tensor,
    override: float | None = None,
    tau_s: float = 0.05,
    target_rad: float = 1e-4,
    floor_s: float = 0.25,
) -> float:
    """Analytic (EMA-only) burn-in floor — kept for reference and overrides.

    Prefer :func:`measure_convergence_burn_in` for franka_fr3 fits: this
    formula under-trims once Ruckig limits dominate (large initial mismatch).
    """
    if override is not None:
        return require_finite("loss_burn_in_s override", override, lo=0.0)
    if shaper.type == "none":
        return 0.0
    m0 = float((dof_pos0 - des0).abs().max())
    if m0 <= target_rad:
        return floor_s
    return max(floor_s, tau_s * math.log(m0 / target_rad))
