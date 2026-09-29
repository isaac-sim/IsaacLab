# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate golden reference outputs for the BAM XL330 (m6) actuator parity tests.

The goldens are produced by the *reference* `BAM <https://github.com/Rhoban/bam>`_
package so that the Isaac Lab actuator port can be checked against it numerically
instead of against a re-derivation of its equations.

Reference version
-----------------
The reference is the ``mjlab_frictionloss`` branch of ``Rhoban/bam`` (NOT ``master``
and NOT the PyPI release of ``better-actuator-models``). Two commits on that branch
matter:

* ``62bd8ce12154340be97e06f7f41a0ca8f116d967`` -- the revision pinned by the
  ``microduck_rl`` consumer's ``uv.lock``. This is what the goldens are generated
  from (see :data:`BAM_COMMIT`).
* ``57d13ead53206a6bf0db3d66f86506ae8c2ce01a`` -- branch head at the time of
  writing (2026-08-28).

Between those two commits the control law (:meth:`VoltageControlledActuator.compute_control`),
the motor-torque equation (:meth:`VoltageControlledActuator.compute_torque`), the
``m6`` friction budget (``BamActuator._compute_friction_budget``) and
``params/xl330/m6.json`` are all unchanged, so the goldens below are identical for
either revision. What *did* change is the supply-voltage sag model, which the goldens
do not cover (they use a fixed ``vin``):

* at ``62bd8ce`` -- ``BamActuatorCfg.vin_drop_gain_range`` [V/Nm]::

      vin_eff = max(vin - vin_drop_gain * sum_j |tau_motor_prev,j|, vin_min)

  where ``tau_motor_prev`` is the previous step's *computed motor torque*
  (cached by the actuator, zeroed on reset).
* at branch head -- ``BamActuatorCfg.vin_drop_resistance_range`` [Ohm]::

      vin_eff = max(vin - R_drop * (sum_j |tau_actuator_prev,j| / kt), vin_min)

  where ``tau_actuator_prev`` is MuJoCo's ``data.qfrc_actuator`` from the previous
  solve. This is the same functional form with ``gain = R_drop / kt``, but the
  torque it reads differs.

Fixture layout
--------------
The ``.npz`` holds three groups of ``float64`` arrays, all of shape ``(1024,)``:

* inputs -- ``q_target`` [rad], ``q`` [rad], ``dq`` [rad/s], ``prev_tau`` [Nm]
  (previous-step motor-side torque), ``ext_tau`` [Nm] (external/gearbox torque).
* reference outputs -- ``duty`` [-], ``volts`` [V], ``motor_torque`` [Nm],
  ``frictionloss_budget`` [Nm], ``stribeck_coeff`` [-].
* scalars -- every key prefixed with ``attr_`` (e.g. ``attr_kt``, ``attr_R``,
  ``attr_max_current``, ``attr_bam_commit``), holding the motor/firmware constants and
  the fitted friction parameters needed to configure the port under test.

Which friction budget
---------------------
The goldens follow ``BamActuator._compute_friction_budget`` (the mjlab/GPU path that
the microduck consumer runs), *not* :meth:`bam.model.Model.compute_frictions` (the
numpy/CPU path). The two agree for m1--m5 but differ on the m6 quadratic term: the CPU
path additionally gates it on ``sign(ext_tau) != sign(motor_torque)`` and splits the
directional masks with strict inequalities, whereas the mjlab path applies the term
unconditionally with ``drive_mask = |motor_torque| > |ext_tau|``. Port the mjlab form.

Usage
-----
.. code-block:: bash

    uv run --with "git+https://github.com/Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967" \
        python scripts/tools/generate_bam_goldens.py

The friction budget is lifted verbatim out of the installed ``bam/mjlab.py`` source
(see :func:`load_reference_friction_budget`) rather than re-implemented here, so the
goldens cannot silently drift from the reference. ``bam.mjlab`` itself is never
imported because it pulls in ``mjlab``/``mujoco_warp``.

The run aborts before generating anything unless the *installed* ``bam`` distribution
really is :data:`BAM_COMMIT` (see :func:`verify_installed_bam_revision`), so the
``attr_bam_commit`` stamped into the fixture can never be a stale claim.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import subprocess
import types
from pathlib import Path
from urllib.request import url2pathname

import numpy as np
import torch
from bam.actuator import TorchBackend, VoltageControlledActuator
from bam.model import Model, load_model

# Reference revision the goldens are generated from (Rhoban/bam @ mjlab_frictionloss).
BAM_COMMIT = "62bd8ce12154340be97e06f7f41a0ca8f116d967"
# Distribution name of the ``bam`` import package on PyPI / in the git repo metadata.
BAM_DIST_NAME = "better-actuator-models"

# Actuator under test: Dynamixel XL330 with the m6 (directional + quadratic) friction model.
MOTOR_NAME = "xl330"
MODEL_NAME = "m6"

# Firmware / supply configuration matching the microduck_rl consumer's nominal setup.
KP_FW = 200.0
VIN = 7.4
# Control timestep [s]. ``compute_control`` ignores it, but it is recorded so a
# consumer of the goldens calls the port with the same value.
DT = 0.005

# Sampling of the input grid.
NUM_SAMPLES = 1024
SEED = 0
Q_RANGE = (-np.pi, np.pi)  # joint angle and target [rad]
DQ_RANGE = (-20.0, 20.0)  # joint velocity [rad/s]
TAU_RANGE = (-1.5, 1.5)  # previous motor torque and external torque [Nm]

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "source/isaaclab/test/actuators/data/bam_xl330_m6_goldens.npz"


def _require(condition: bool, message: str) -> None:
    """Raise :class:`RuntimeError` when ``condition`` is false.

    Used instead of ``assert`` so the checks still run under ``python -O``: a golden
    fixture that silently skipped its own validation would be worse than no fixture.
    """
    if not condition:
        raise RuntimeError(message)


def verify_installed_bam_revision() -> None:
    """Abort unless the installed ``bam`` distribution is exactly :data:`BAM_COMMIT`.

    ``attr_bam_commit`` is written into the fixture from the :data:`BAM_COMMIT` constant, so
    without this guard a run against a drifted revision would stamp the fixture with a SHA that
    did not produce it. A local checkout is deliberately *not* accepted here: the fixture is
    committed and its provenance stamp has to name a published commit.

    Raises:
        RuntimeError: If ``bam`` is missing, was not installed from git (e.g. an editable path
            or a PyPI wheel, neither of which carries a commit id), or was installed from a
            different commit than :data:`BAM_COMMIT`.
    """
    resolve_installed_bam_revision(allow_local_checkout=False)


def resolve_installed_bam_revision(*, allow_local_checkout: bool = False) -> str:
    """Return the revision of the installed ``bam``, or raise if it is not :data:`BAM_COMMIT`.

    ``pip``/``uv`` record where a distribution came from in ``direct_url.json`` (PEP 610). A git
    install carries the resolved commit directly; a local checkout carries only its path, so its
    ``git`` HEAD is read instead and the working tree must be clean -- otherwise the reported
    revision would not describe the code that actually ran.

    This is the single home of the revision policy. The goldens generator stamps a committed
    fixture and therefore refuses anything but a git install; ``bam_parity_rollout.py`` reports
    numbers it prints alongside the revision, so the pinned local checkout the parity runs use
    is accepted there.

    Args:
        allow_local_checkout: Whether a ``file://`` install whose git HEAD is :data:`BAM_COMMIT`
            and whose working tree is clean counts as verified.

    Returns:
        The verified commit id of the installed ``bam``.

    Raises:
        RuntimeError: If ``bam`` is missing, its revision cannot be established, or it differs
            from :data:`BAM_COMMIT`.
    """
    try:
        distribution = importlib.metadata.distribution(BAM_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(f"{BAM_DIST_NAME!r} is not installed; see the Usage section of this module's docstring.")

    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError(
            f"{BAM_DIST_NAME!r} has no direct_url.json, so its revision cannot be verified against the expected"
            f" {BAM_COMMIT}. Install it from git as shown in the Usage section of this module's docstring."
        )
    metadata = json.loads(direct_url)
    installed_commit = metadata.get("vcs_info", {}).get("commit_id")
    if installed_commit is None:
        if not allow_local_checkout:
            raise RuntimeError(
                f"{BAM_DIST_NAME!r} was not installed from a VCS (direct_url.json carries no vcs_info.commit_id), so"
                f" its revision cannot be verified against the expected {BAM_COMMIT}."
            )
        installed_commit = _revision_of_local_checkout(metadata.get("url", ""))

    _require(
        installed_commit == BAM_COMMIT,
        f"Installed {BAM_DIST_NAME!r} revision does not match the reference this tool is pinned to.\n"
        f"  expected (BAM_COMMIT): {BAM_COMMIT}\n"
        f"  installed:             {installed_commit}\n"
        "Either re-run with the expected revision, or update BAM_COMMIT (and this module's docstring) after"
        " re-verifying that the reference equations still match.",
    )
    return installed_commit


def _revision_of_local_checkout(url: str) -> str:
    """Return the git HEAD of a ``file://`` install, requiring a clean working tree."""
    if not url.startswith("file://"):
        raise RuntimeError(
            f"{BAM_DIST_NAME!r} was installed from {url!r}, which carries neither a commit id nor a local path, so"
            f" its revision cannot be verified against the expected {BAM_COMMIT}."
        )
    checkout = Path(url2pathname(url[len("file://") :])).resolve()

    def _git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)  # noqa: S603, S607
        _require(
            result.returncode == 0,
            f"Could not read the git state of the {BAM_DIST_NAME!r} checkout at {checkout}: {result.stderr.strip()!r}.",
        )
        return result.stdout.strip()

    _require(
        Path(_git("rev-parse", "--show-toplevel")).resolve() == checkout,
        f"{checkout} is not the root of a git repository, so it cannot identify a {BAM_DIST_NAME!r} revision.",
    )
    _require(
        _git("status", "--porcelain") == "",
        f"The {BAM_DIST_NAME!r} checkout at {checkout} has uncommitted changes, so its HEAD does not describe the"
        " code that would run. Commit or stash them first.",
    )
    return _git("rev-parse", "HEAD")


def load_reference_friction_budget():
    """Extract ``BamActuator._compute_friction_budget`` from the installed ``bam.mjlab`` source.

    ``bam.mjlab`` imports ``mjlab`` and ``mujoco_warp``, neither of which is needed to
    evaluate the friction budget: the method only touches ``torch`` and the BAM
    :class:`~bam.model.Model` it is bound to. The function definition is therefore
    parsed out of the module source and compiled in isolation, which keeps the goldens
    tied to the reference implementation byte-for-byte.

    Returns:
        The unbound ``_compute_friction_budget`` function. Call it as
        ``fn(shim, motor_torque, external_torque, stribeck_coeff)`` where ``shim`` is any
        object exposing ``_bam_model``.
    """
    spec = importlib.util.find_spec("bam.mjlab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate bam/mjlab.py in the installed bam package.")
    source_path = Path(spec.origin)
    tree = ast.parse(source_path.read_text())

    def _find(nodes, node_type, name):
        for node in nodes:
            if isinstance(node, node_type) and node.name == name:
                return node
        raise RuntimeError(f"{name} not found in {source_path}; the reference layout changed.")

    cls = _find(tree.body, ast.ClassDef, "BamActuator")
    fn = _find(cls.body, ast.FunctionDef, "_compute_friction_budget")

    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"torch": torch}
    exec(compile(module, filename=str(source_path), mode="exec"), namespace)  # noqa: S102
    return namespace["_compute_friction_budget"]


def build_reference_model() -> Model:
    """Load the BAM ``xl330``/``m6`` model configured like the microduck_rl consumer."""
    model = load_model(motor_name=MOTOR_NAME, model=MODEL_NAME)
    actuator = model.actuator
    if not isinstance(actuator, VoltageControlledActuator):
        raise RuntimeError(f"Expected a VoltageControlledActuator, got {type(actuator).__name__}.")
    actuator.vin = VIN
    actuator.kp = KP_FW
    # Torch backend so the clamps vectorize over the sample batch, exactly as BamActuator does.
    actuator.backend = TorchBackend()
    return model


def sample_inputs() -> dict[str, np.ndarray]:
    """Draw the reproducible input grid. NumPy's PCG64 keeps this stable across platforms."""
    rng = np.random.default_rng(SEED)
    return {
        "q_target": rng.uniform(*Q_RANGE, NUM_SAMPLES),
        "q": rng.uniform(*Q_RANGE, NUM_SAMPLES),
        "dq": rng.uniform(*DQ_RANGE, NUM_SAMPLES),
        "prev_tau": rng.uniform(*TAU_RANGE, NUM_SAMPLES),
        "ext_tau": rng.uniform(*TAU_RANGE, NUM_SAMPLES),
    }


def compute_goldens(model: Model, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the reference pipeline over the sampled inputs.

    The three stages mirror ``BamActuator.compute``: firmware control law, DC-motor
    torque equation, and the velocity-independent friction budget. Unlike ``compute``,
    the motor-side and external torques fed to the budget are explicit inputs rather
    than MuJoCo state, so the goldens need no simulator.
    """
    actuator = model.actuator
    tensors = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in inputs.items()}

    # 1. Firmware P-controller (+ current limiter, + PWM clamp) -> volts.
    volts = actuator.compute_control(tensors["q_target"], tensors["q"], tensors["dq"], DT)
    # ``compute_control`` returns ``vin * duty_cycle``; recover the duty it clamped to.
    duty = volts / VIN

    # 2. DC-motor equation with back-EMF -> motor torque [Nm].
    motor_torque = actuator.compute_torque(volts, True, tensors["q"], tensors["dq"])

    # 3. Stribeck coefficient (1 at rest, 0 when moving), as computed in BamActuator.compute.
    stribeck_coeff = torch.exp(-torch.pow(torch.abs(tensors["dq"]) / model.dtheta_stribeck.value, model.alpha.value))

    # 4. m6 friction budget from the reference implementation.
    friction_budget_fn = load_reference_friction_budget()
    shim = types.SimpleNamespace(_bam_model=model)
    frictionloss_budget = friction_budget_fn(shim, tensors["prev_tau"], tensors["ext_tau"], stribeck_coeff)

    return {
        "duty": duty.numpy(),
        "volts": volts.numpy(),
        "motor_torque": motor_torque.numpy(),
        "frictionloss_budget": frictionloss_budget.numpy(),
        "stribeck_coeff": stribeck_coeff.numpy(),
    }


def collect_scalars(model: Model) -> dict[str, float | str | int]:
    """Collect the model/firmware scalars a consumer needs to configure the port."""
    actuator = model.actuator
    scalars: dict[str, float | str | int] = {
        "bam_commit": BAM_COMMIT,
        "motor_name": MOTOR_NAME,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        # Firmware / supply.
        "kp": KP_FW,
        "vin": VIN,
        "dt": DT,
        "error_gain": actuator.error_gain,
        "max_pwm": actuator.max_pwm,
        "max_current": actuator.max_current,
    }
    # Every fitted parameter of the model (kt, R, armature, friction terms, ...).
    scalars.update({name: param.value for name, param in model.get_parameters().items()})
    return scalars


def check_stribeck_against_reference(
    model: Model, inputs: dict[str, np.ndarray], goldens: dict[str, np.ndarray]
) -> None:
    """Tripwire: check the inline Stribeck re-derivation against the reference.

    The Stribeck coefficient is the one quantity :func:`compute_goldens` re-derives by hand,
    because upstream computes it inline in ``BamActuator.compute`` rather than in an
    extractable helper. :meth:`bam.model.Model.compute_frictions` computes the same
    coefficient internally, and for the m6 model it can be read back out exactly: with
    ``motor_torque = external_torque = 0`` every load-dependent and quadratic term vanishes,
    leaving ``frictionloss = friction_base + stribeck_coeff * friction_stribeck``.

    So if upstream ever changes the Stribeck law, this fails instead of silently baking a
    stale formula into the fixture.
    """
    friction_base = model.friction_base.value
    friction_stribeck = model.friction_stribeck.value
    # A spread of velocities rather than a contiguous slice: the coefficient underflows to
    # zero well before |dq| = 20, so neighbouring samples would not exercise the curve.
    probe_indices = np.argsort(np.abs(inputs["dq"]))[:: max(1, NUM_SAMPLES // 16)]
    for index in probe_indices:
        frictionloss, _ = model.compute_frictions(0.0, 0.0, inputs["dq"][index])
        reference_coeff = (frictionloss - friction_base) / friction_stribeck
        _require(
            np.isclose(goldens["stribeck_coeff"][index], reference_coeff, rtol=1e-12, atol=1e-15),
            f"Stribeck re-derivation drifted from the reference at sample {index}"
            f" (dq={inputs['dq'][index]:.6f}): got {goldens['stribeck_coeff'][index]!r},"
            f" reference gives {reference_coeff!r}.",
        )


def check_goldens(scalars: dict, inputs: dict[str, np.ndarray], goldens: dict[str, np.ndarray]) -> None:
    """Sanity-check the generated goldens against independently derived bounds."""
    kt = scalars["kt"]
    resistance = scalars["R"]
    max_current = scalars["max_current"]

    # Duty cycle can never leave the physical PWM range (clamped last in compute_control).
    _require(np.all(np.abs(goldens["duty"]) <= scalars["max_pwm"] + 1e-12), "duty exceeds max_pwm")
    _require(np.allclose(goldens["volts"], goldens["duty"] * VIN), "volts != duty * vin")

    # Motor torque is kt * I. The firmware limiter targets |I| <= max_current but can only
    # clamp the duty, so at high back-EMF the PWM rail wins: |I| <= (vin + kt*|dq|max) / R.
    max_speed = max(abs(DQ_RANGE[0]), abs(DQ_RANGE[1]))
    torque_bound = kt * max(max_current, (VIN + kt * max_speed) / resistance)
    _require(np.all(np.abs(goldens["motor_torque"]) <= torque_bound + 1e-9), "motor torque exceeds the DC-motor bound")

    # Every friction term added on top of friction_base is non-negative.
    _require(np.all(goldens["frictionloss_budget"] >= scalars["friction_base"] - 1e-12), "budget below friction_base")

    # Stribeck coefficient is a decaying exponential in |dq|.
    _require(np.all((goldens["stribeck_coeff"] >= 0.0) & (goldens["stribeck_coeff"] <= 1.0)), "stribeck outside [0, 1]")

    # Independent recomputation of the torque equation (tau = kt*V/R - kt^2*dq/R).
    expected_torque = kt * goldens["volts"] / resistance - (kt**2) * inputs["dq"] / resistance
    _require(np.allclose(goldens["motor_torque"], expected_torque), "motor torque does not match the DC-motor equation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path of the .npz fixture to write.")
    args = parser.parse_args()

    verify_installed_bam_revision()
    model = build_reference_model()
    inputs = sample_inputs()
    goldens = compute_goldens(model, inputs)
    scalars = collect_scalars(model)
    check_goldens(scalars, inputs, goldens)
    check_stribeck_against_reference(model, inputs, goldens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **inputs, **goldens, **{f"attr_{k}": np.asarray(v) for k, v in scalars.items()})

    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"bam commit: {BAM_COMMIT}  motor: {MOTOR_NAME}/{MODEL_NAME}  kp={KP_FW}  vin={VIN}")
    for name, array in {**inputs, **goldens}.items():
        print(
            f"  {name:20s} shape={array.shape} min={array.min(): .6f} max={array.max(): .6f} mean={array.mean(): .6f}"
        )


if __name__ == "__main__":
    main()
