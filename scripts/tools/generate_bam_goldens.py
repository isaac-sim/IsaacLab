# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate XL330/m6 reference samples from the pinned upstream BAM implementation.

Usage::

    uv run --with "git+https://github.com/Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967" \
        --with "mjlab==1.3.0" python scripts/tools/generate_bam_goldens.py

The fixture stores float64 input/output arrays and ``attr_`` metadata for attribution, revision,
sampling settings, firmware constants and fitted parameters. Inputs are positions [rad],
velocities [rad/s] and torques [N.m]; outputs are duty cycles [-], voltages [V], motor
torques [N.m], friction budgets [N.m] and Stribeck coefficients [-]. Additional recordings
cover battery sag and mjlab 1.3.0 command delays of 3--6 steps, including a partial reset.

Friction uses upstream's mjlab method, whose m6 quadratic term differs from its CPU model.
Extracting that method avoids importing the mjlab simulator or duplicating its equations.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from bam.actuator import TorchBackend
from bam.model import Model, load_model

BAM_COMMIT = "62bd8ce12154340be97e06f7f41a0ca8f116d967"
MOTOR_NAME = "xl330"
MODEL_NAME = "m6"
KP_FW = 200.0
VIN = 7.4  # Supply voltage [V].
DT = 0.005  # Control timestep [s].
NUM_SAMPLES = 1024
SEED = 0
Q_RANGE = (-np.pi, np.pi)  # Joint angle and target [rad].
DQ_RANGE = (-20.0, 20.0)  # Joint velocity [rad/s].
TAU_RANGE = (-1.5, 1.5)  # Previous motor and external torques [N.m].
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "source/isaaclab/test/actuators/data/bam_xl330_m6_goldens.npz"


def verify_installed_bam_revision() -> None:
    """Require the pinned git installation before stamping fixture provenance."""
    distribution = importlib.metadata.distribution("better-actuator-models")
    metadata = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed_commit = metadata.get("vcs_info", {}).get("commit_id")
    if installed_commit != BAM_COMMIT:
        raise RuntimeError(
            f"Expected BAM git revision {BAM_COMMIT}, got {installed_commit!r}."
            " Install the pinned revision using the command in this script's docstring."
        )


def load_reference(source_path: Path, class_name: str, method: str | None = None, **bindings) -> Callable:
    """Execute an upstream class or method unchanged, without importing simulator dependencies."""
    tree = ast.parse(source_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition = (
        next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method) if method else cls
    )
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = {"torch": torch, "Sequence": Sequence, "ActuatorCmd": SimpleNamespace, **bindings}
    exec(compile(ast.fix_missing_locations(module), filename=str(source_path), mode="exec"), namespace)  # noqa: S102
    return namespace[method or class_name]


def load_bam_method(method: str) -> Callable:
    """Load a method from the verified BAM installation."""
    spec = importlib.util.find_spec("bam.mjlab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate bam/mjlab.py.")
    return load_reference(Path(spec.origin), "BamActuator", method)


def sample_inputs() -> dict[str, np.ndarray]:
    """Draw the reference input grid with a fixed seed and sampling order."""
    rng = np.random.default_rng(SEED)
    return {
        "q_target": rng.uniform(*Q_RANGE, NUM_SAMPLES),
        "q": rng.uniform(*Q_RANGE, NUM_SAMPLES),
        "dq": rng.uniform(*DQ_RANGE, NUM_SAMPLES),
        "prev_tau": rng.uniform(*TAU_RANGE, NUM_SAMPLES),
        "ext_tau": rng.uniform(*TAU_RANGE, NUM_SAMPLES),
    }


def compute_goldens(model: Model, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate upstream firmware control, motor torque and gearbox friction."""
    actuator = model.actuator
    tensors = {name: torch.as_tensor(value, dtype=torch.float64) for name, value in inputs.items()}
    volts = actuator.compute_control(tensors["q_target"], tensors["q"], tensors["dq"], DT)
    motor_torque = actuator.compute_torque(volts, True, tensors["q"], tensors["dq"])
    # Upstream computes this coefficient inline in BamActuator.compute.
    stribeck_coeff = torch.exp(-torch.pow(torch.abs(tensors["dq"]) / model.dtheta_stribeck.value, model.alpha.value))
    frictionloss_budget = load_bam_method("_compute_friction_budget")(
        SimpleNamespace(_bam_model=model), tensors["prev_tau"], tensors["ext_tau"], stribeck_coeff
    )
    return {
        "duty": (volts / VIN).numpy(),
        "volts": volts.numpy(),
        "motor_torque": motor_torque.numpy(),
        "frictionloss_budget": frictionloss_budget.numpy(),
        "stribeck_coeff": stribeck_coeff.numpy(),
    }


def record_supply_sag(model: Model) -> dict[str, np.ndarray]:
    """Record BAM.compute with supplied joint states and zero solver loads, retaining torque history."""
    rng = np.random.default_rng(SEED)
    shape = (32, 3, 2)  # Steps, environments, joints.
    inputs = {
        "q": rng.uniform(-0.4, 0.4, shape),
        "dq": rng.uniform(-60.0, 60.0, shape),
        "target": rng.uniform(-0.4, 0.4, shape),
    }
    zeros = torch.zeros(shape[1:], dtype=torch.float64)
    reference = SimpleNamespace(
        _bam_model=model,
        _base_kp=KP_FW,
        _dt=DT,
        vin_tensor=torch.tensor([[6.5], [7.4], [8.2]], dtype=torch.float64),
        vin_drop_gain=torch.tensor([[0.2], [0.0], [5.0]], dtype=torch.float64),
        kp_scale=torch.ones((3, 1)),
        kd_scale=torch.ones((3, 1)),
        cfg=SimpleNamespace(vin_min=6.0),
        _prev_motor_torque=zeros.clone(),
        _dof_ids=torch.arange(2),
        _mjwarp_model=object(),
        _data=SimpleNamespace(qfrc_bias=zeros, qfrc_constraint=zeros, qfrc_actuator=zeros),
        _as_tensor=lambda value: value,
        _dof_friction_force=lambda nv: zeros,
        _write_frictions=lambda budget, viscous: None,
    )
    reference._compute_friction_budget = MethodType(load_bam_method("_compute_friction_budget"), reference)
    compute = load_bam_method("compute")
    voltages, torques = [], []
    for q, dq, target in zip(*inputs.values(), strict=True):
        command = SimpleNamespace(
            pos=torch.from_numpy(q), vel=torch.from_numpy(dq), position_target=torch.from_numpy(target)
        )
        torques.append(compute(reference, command).numpy().copy())
        voltages.append(model.actuator.vin.numpy().copy())
    return {
        **{f"sag_{key}": value for key, value in inputs.items()},
        "sag_vin": reference.vin_tensor.numpy(),
        "sag_gain": reference.vin_drop_gain.numpy(),
        "sag_vin_min": np.asarray(reference.cfg.vin_min),
        "sag_effective_vin": np.asarray(voltages),
        "sag_motor_torque": np.asarray(torques),
    }


def record_delay(model: Model) -> dict[str, np.ndarray]:
    """Record mjlab's actual lag draws and delayed commands, plus upstream BAM motor outputs."""
    distribution = importlib.metadata.distribution("mjlab")
    if distribution.version != "1.3.0":
        raise RuntimeError(f"Expected mjlab 1.3.0, got {distribution.version}")
    root = Path(distribution.locate_file("mjlab/utils/buffers"))
    circular = load_reference(root / "circular_buffer.py", "CircularBuffer")
    delay_type = load_reference(root / "delay_buffer.py", "DelayBuffer", CircularBuffer=circular)
    buffer = delay_type(min_lag=3, max_lag=6, batch_size=2, generator=torch.Generator().manual_seed(SEED))
    commands = np.random.default_rng(SEED).uniform(-0.03, 0.03, (32, 2, 2))
    resets = np.zeros((32, 2), dtype=bool)
    resets[15, 0] = True
    model.actuator.vin, model.actuator.kp = VIN, KP_FW
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    lags, delayed, torques = [], [], []
    for command, reset in zip(commands, resets, strict=True):
        if reset.any():
            buffer.reset(np.flatnonzero(reset).tolist())
        buffer.append(torch.from_numpy(command))
        target = buffer.compute()
        lags.append(buffer.current_lags.numpy().copy())
        delayed.append(target.numpy().copy())
        volts = model.actuator.compute_control(target, zeros, zeros, DT)
        torques.append(model.actuator.compute_torque(volts, True, zeros, zeros).numpy().copy())
    return {
        "delay_commands": commands,
        "delay_resets": resets,
        "delay_lags": np.asarray(lags),
        "delay_targets": np.asarray(delayed),
        "delay_motor_torque": np.asarray(torques),
        "delay_min": np.asarray(3),
        "delay_max": np.asarray(6),
        "delay_reference": np.asarray("mjlab 1.3.0; https://github.com/mujocolab/mjlab; utils/buffers/DelayBuffer"),
        "delay_torch_version": np.asarray(torch.__version__),
        "delay_source_sha256": np.asarray(
            [
                hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("delay_buffer.py", "circular_buffer.py")
            ]
        ),
    }


def collect_scalars(model: Model) -> dict[str, float | str | int]:
    """Collect provenance, firmware constants and fitted motor/friction parameters."""
    actuator = model.actuator
    return {
        "bam_commit": BAM_COMMIT,
        "bam_attribution": (
            "BAM (Better Actuator Models) by Marc Duclusaud and Grégoire Passault; https://github.com/Rhoban/bam. "
            f"Fit: bam/params/{MOTOR_NAME}/{MODEL_NAME}.json; firmware: bam/dynamixel/actuator.py (XL330Actuator)."
        ),
        "motor_name": MOTOR_NAME,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        "kp": KP_FW,
        "vin": VIN,
        "dt": DT,
        "error_gain": actuator.error_gain,
        "max_pwm": actuator.max_pwm,
        "max_current": actuator.max_current,
        **{name: param.value for name, param in model.get_parameters().items()},
    }


def main() -> None:
    """Generate the reference fixture at the requested output path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path of the .npz fixture to write.")
    args = parser.parse_args()

    verify_installed_bam_revision()
    model = load_model(motor_name=MOTOR_NAME, model=MODEL_NAME)
    model.actuator.vin = VIN
    model.actuator.kp = KP_FW
    model.actuator.backend = TorchBackend()
    inputs = sample_inputs()
    goldens = compute_goldens(model, inputs)
    for name, values in goldens.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite reference output: {name}")
    scalars = collect_scalars(model)
    temporal = {**record_supply_sag(model), **record_delay(model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **inputs,
        **goldens,
        **temporal,
        **{f"attr_{key}": np.asarray(value) for key, value in scalars.items()},
    )
    print(f"Wrote {args.output} ({NUM_SAMPLES} samples, BAM {BAM_COMMIT})")


if __name__ == "__main__":
    main()
