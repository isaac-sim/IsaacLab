# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task integration helpers shared with the dynamics recording and fitting script tests.

Importing this module needs neither ``newton-neural`` nor a running simulation. The helpers that need them import
them when called.
"""

from __future__ import annotations

import contextlib
import logging
import runpy
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import pytest

from isaaclab.app import launch_simulation
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

if TYPE_CHECKING:
    import newton

FRANKA_REACH = "Isaac-Reach-Franka"
ANYMAL_C_FLAT = "IsaacContrib-Velocity-Flat-AnymalC"
ALLEGRO_REORIENT = "Isaac-Reorient-Cube-Allegro"

TASKS = (FRANKA_REACH, ANYMAL_C_FLAT, ALLEGRO_REORIENT)
NEURAL_PHYSICS = {
    FRANKA_REACH: "newton_nerd",
    ANYMAL_C_FLAT: "newton_nerd",
    ALLEGRO_REORIENT: "newton_mjwarp_nerd",
}
"""The tasks that run on a model."""

WORLD_COUNT = 32
"""Environments of a task that a test steps."""


def cuda_device() -> str:
    """Return the first CUDA test device, or skip the test if there is none."""
    devices = test_devices(DeviceScope.CUDA)
    if not devices:
        pytest.skip("the models are recorded and trained on a CUDA device")
    return devices[0]


def task_cfg(task: str, world_count: int, device: str, overrides: tuple[str, ...] | None = None) -> Any:
    """Resolve ``task`` for ``world_count`` environments on ``device``, on MuJoCo Warp unless ``overrides`` say so."""
    overrides = ("physics=newton_mjwarp",) if overrides is None else overrides
    env_cfg, _ = resolve_task_config(task, None, overrides=overrides)
    env_cfg.scene.num_envs = world_count
    env_cfg.sim.device = device
    env_cfg.seed = 0
    return env_cfg


def nerd_task_cfg(task: str, checkpoint: Path, world_count: int, device: str, **solver_settings: Any) -> Any:
    """Resolve ``task`` on the physics preset of its model, with the model at ``checkpoint``.

    Args:
        task: Task to resolve.
        checkpoint: File of the trained model.
        world_count: Number of environments.
        device: Simulation device.
        **solver_settings: Fields of the solver configuration to set, such as ``precision`` and ``compile``.
    """
    env_cfg = task_cfg(task, world_count, device, overrides=(f"physics={NEURAL_PHYSICS[task]}",))
    solver_cfg = env_cfg.sim.physics.solver_cfg
    solver_cfg.checkpoint = str(checkpoint)
    for name, value in solver_settings.items():
        setattr(solver_cfg, name, value)
    return env_cfg


@contextlib.contextmanager
def simulation(cfg: Any, *, info: bool = False) -> Iterator[None]:
    """Launch the headless simulation of ``cfg`` and clear the ``SimulationContext`` on exit."""
    try:
        with launch_simulation(cfg, {"device": cfg.sim.device, "headless": True, "info": info}):
            yield
    finally:
        SimulationContext.clear_instance()


@contextlib.contextmanager
def environment(task: str, env_cfg: Any, *, info: bool = False) -> Iterator[Any]:
    """Create the unwrapped headless environment of ``task`` and close it on exit."""
    env = None
    with simulation(env_cfg, info=info):
        try:
            env = gym.make(task, cfg=env_cfg).unwrapped
            yield env
        finally:
            if env is not None:
                env.close()


def refused(task: str, env_cfg: Any, error: type[Exception], match: str) -> None:
    """Expect the environment of ``task`` on ``env_cfg`` to be refused with ``error`` and a message like ``match``."""
    with pytest.raises(error, match=match), environment(task, env_cfg):
        pass


def record_dataset(task: str, dataset: Path, num_envs: int, num_frames: int, device: str, *overrides: str) -> None:
    """Record a dataset of ``task`` with ``scripts/neural/record.py``, in ``dataset`` or where the script puts it."""
    from isaaclab.paths import ISAACLAB_ROOT  # noqa: PLC0415

    argv = ["--task", task, "--num_envs", str(num_envs), "--num_frames", str(num_frames), "--device", device]
    argv += ["--dataset", str(dataset), *overrides]
    try:
        runpy.run_path(str(Path(ISAACLAB_ROOT) / "scripts/neural/record.py"))["main"](argv)
    finally:
        SimulationContext.clear_instance()


def record_and_train(task: str, folder: Path, *, num_envs: int, num_frames: int, max_steps: int) -> Path:
    """Record a small dataset of ``task`` and train its model on it, with the scripts under ``scripts/neural``.

    Args:
        task: Task declaring the model, recording environment and optimization settings.
        folder: Folder that receives the dataset and the model.
        num_envs: Robots recorded in parallel.
        num_frames: Frames recorded per robot.
        max_steps: Optimizer steps.

    Returns:
        The trained model file. The adjacent ``dataset`` folder holds the native trajectories.
    """
    from isaaclab.paths import ISAACLAB_ROOT  # noqa: PLC0415

    device = cuda_device()
    dataset, checkpoint = folder / "dataset", folder / "flashnerd.pt"
    record_dataset(task, dataset, num_envs, num_frames, device)
    runpy.run_path(str(Path(ISAACLAB_ROOT) / "scripts/neural/train.py"))["main"](
        [
            *("--task", task),
            *("--dataset", str(dataset)),
            *("--checkpoint", str(checkpoint)),
            *("--max_steps", str(max_steps)),
            *("--device", device),
        ]
    )
    return checkpoint


def save_point_model(path: Path) -> Path:
    """Save at ``path`` the model of a point that moves, a system that is not an articulated robot, and return it."""
    from newton_neural import Frames, SignalRole, SignalSpec  # noqa: PLC0415
    from newton_neural.backbones import BackboneMLP  # noqa: PLC0415
    from newton_neural.models import FlashNeRD  # noqa: PLC0415
    from newton_neural.systems import SystemBase  # noqa: PLC0415
    from torch import Tensor  # noqa: PLC0415

    class SystemPoint(SystemBase):
        """A point with its position as the state. It binds to no Newton model, so no solver can be built on it."""

        def __init__(self) -> None:
            super().__init__()
            self.signals = {"position": SignalSpec("position", (3,), SignalRole.STATE, unit="m")}
            self.feature_count = self.increment_count = 3

        def features(self, frames: Frames) -> Tensor:
            return frames["position"]

        def increment(self, current: Frames, following: Frames) -> Tensor:
            return following["position"] - current["position"]

        def integrate(self, current: Frames, increment: Tensor) -> Frames:
            return {"position": current["position"] + increment}

        def config(self) -> dict[str, Any]:
            return {}

    FlashNeRD(SystemPoint(), backbone=BackboneMLP(width=16, depth=1, window=1), frame_dt=0.02).save(path)
    return path


@contextlib.contextmanager
def recorded_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Collect every warning raised within the block, repeated ones included."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def unseen_changes(caught: list[warnings.WarningMessage]) -> list[str]:
    """Return the messages of the ``NewtonNeuralWarning`` entries of ``caught``."""
    from newton_neural import NewtonNeuralWarning  # noqa: PLC0415

    return [str(warning.message) for warning in caught if issubclass(warning.category, NewtonNeuralWarning)]


def notify_change(env: Any, change: newton.ModelFlags, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Tell the solver of ``env`` about ``change`` and return what it warns about.

    Asserts that Isaac Lab logs no warning of its own about the change.
    """
    env.sim.backend_manager.add_model_change(change)
    caplog.clear()
    with recorded_warnings() as caught, caplog.at_level(logging.WARNING, logger="isaaclab_newton"):
        env.sim.backend_manager.backend.notify_model_changes()
    assert [entry for entry in caplog.records if entry.name.split(".")[0] == "isaaclab_newton"] == []
    return unseen_changes(caught)
