# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate XL330/m6 reference samples from the pinned upstream BAM implementation.

Usage::

    uv run --with "git+https://github.com/Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967" \
        python scripts/tools/generate_bam_goldens.py

The fixture stores float64 input/output arrays and ``attr_`` metadata for the revision,
sampling settings, firmware constants and fitted parameters. Inputs are positions [rad],
velocities [rad/s] and torques [N.m]; outputs are duty cycles [-], voltages [V], motor
torques [N.m], friction budgets [N.m] and Stribeck coefficients [-]. Supply voltage is fixed.

Friction uses upstream's mjlab method, whose m6 quadratic term differs from its CPU model.
Extracting that method avoids importing the mjlab simulator or duplicating its equations.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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


def load_reference_friction_budget() -> Callable:
    """Extract upstream's friction method without importing its simulator dependencies."""
    spec = importlib.util.find_spec("bam.mjlab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate bam/mjlab.py.")
    source_path = Path(spec.origin)
    tree = ast.parse(source_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BamActuator")
    fn = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_friction_budget"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), filename=str(source_path), mode="exec"), namespace)  # noqa: S102
    return namespace["_compute_friction_budget"]


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
    frictionloss_budget = load_reference_friction_budget()(
        SimpleNamespace(_bam_model=model), tensors["prev_tau"], tensors["ext_tau"], stribeck_coeff
    )
    return {
        "duty": (volts / VIN).numpy(),
        "volts": volts.numpy(),
        "motor_torque": motor_torque.numpy(),
        "frictionloss_budget": frictionloss_budget.numpy(),
        "stribeck_coeff": stribeck_coeff.numpy(),
    }


def collect_scalars(model: Model) -> dict[str, float | str | int]:
    """Collect provenance, firmware constants and fitted motor/friction parameters."""
    actuator = model.actuator
    return {
        "bam_commit": BAM_COMMIT,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **inputs, **goldens, **{f"attr_{key}": np.asarray(value) for key, value in scalars.items()}
    )
    print(f"Wrote {args.output} ({NUM_SAMPLES} samples, BAM {BAM_COMMIT})")


if __name__ == "__main__":
    main()
