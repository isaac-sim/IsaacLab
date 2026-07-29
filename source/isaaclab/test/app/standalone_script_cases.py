# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discovery and subprocess helpers for standalone script smoke tests."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOTS = (ROOT / "scripts" / "demos", ROOT / "scripts" / "tutorials")
VISUALIZERS = ("none", "kit", "newton", "rerun", "viser")
DEFAULT_READINESS_PATTERN = r"Setup complete"
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
DEFAULT_BATCHED_NUM_ENVS = 2

_FATAL_PATTERNS = (
    "Traceback (most recent call last):",
    "Segmentation fault",
    "Fatal Python error:",
    "CUDA error:",
    "exceeded MJWarp limit",
    "nefc overflow",
)


@dataclass(frozen=True)
class ScriptOverride:
    """Describe behavior that cannot be inferred from a standalone script."""

    args: tuple[str, ...] = ()
    readiness_pattern: str | None = None
    startup_timeout: float | None = None
    skip_reason: str | None = None
    fixed_physics_backend: str | None = None
    visualizers: tuple[str, ...] | None = None
    case_skip_reasons: dict[tuple[str, str, str], str] = field(default_factory=dict)
    required_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScriptSpec:
    """Static launch contract for one standalone script."""

    path: Path
    options: dict[str, tuple[str, ...]]
    args: tuple[str, ...]
    readiness_pattern: str | None
    startup_timeout: float | None
    skip_reason: str | None
    fixed_physics_backend: str | None
    visualizers: tuple[str, ...]
    case_skip_reasons: dict[tuple[str, str, str], str]
    visualizer_option: str
    required_modules: tuple[str, ...]

    @property
    def relative_path(self) -> str:
        """Return the repository-relative POSIX path."""
        return self.path.relative_to(ROOT).as_posix()

    @property
    def physics_backends(self) -> tuple[tuple[str | None, str], ...]:
        """Return the script's declared physics backend selections."""
        for option in ("--physics", "--backend"):
            if choices := self.options.get(option):
                return tuple((option, choice) for choice in choices)
        return ((None, self.fixed_physics_backend or "isaacsim_physx"),)

    @property
    def rendering_backends(self) -> tuple[tuple[str | None, str], ...]:
        """Return the script's declared rendering backend selections."""
        if choices := self.options.get("--renderer"):
            return tuple(("--renderer", choice) for choice in choices)
        return ((None, "default"),)


@dataclass(frozen=True)
class LaunchCase:
    """One script, physics backend, renderer, and visualizer combination."""

    spec: ScriptSpec
    physics_option: str | None
    physics_backend: str
    renderer_option: str | None
    renderer_backend: str
    visualizer: str

    @property
    def id(self) -> str:
        """Return a stable pytest identifier."""
        stem = self.spec.relative_path.removeprefix("scripts/").removesuffix(".py").replace("/", "-")
        return f"{stem}-{self.physics_backend}-{self.renderer_backend}-{self.visualizer}"

    @property
    def skip_reason(self) -> str | None:
        """Return the static reason this launch combination cannot run."""
        key = (self.physics_backend, self.renderer_backend, self.visualizer)
        return self.spec.skip_reason or self.spec.case_skip_reasons.get(key)

    def command(self) -> list[str]:
        """Build the repository launcher command for this case."""
        command = [str(ROOT / "isaaclab.sh"), "-p", self.spec.relative_path, *self.spec.args]
        if "--num_envs" in self.spec.options and "--num_envs" not in self.spec.args:
            command.extend(("--num_envs", str(DEFAULT_BATCHED_NUM_ENVS)))
        if self.physics_option is not None:
            command.extend((self.physics_option, self.physics_backend))
        if self.renderer_option is not None:
            command.extend((self.renderer_option, self.renderer_backend))
        command.extend((self.spec.visualizer_option, self.visualizer))
        return command


@dataclass(frozen=True)
class SmokeResult:
    """Result of supervising a standalone script."""

    ready: bool
    returncode: int | None
    output: str
    elapsed: float
    stopped_after_soak: bool
    fatal_patterns: tuple[str, ...] = ()


OVERRIDES = {
    "scripts/demos/arl_robot_1.py": ScriptOverride(readiness_pattern=r"Starting demo with Lee Position Controller"),
    "scripts/demos/h1_locomotion.py": ScriptOverride(
        skip_reason="downloads a published policy and requires interactive viewport input",
        visualizers=("kit",),
    ),
    "scripts/demos/haply_teleoperation.py": ScriptOverride(
        skip_reason="requires a physical Haply device and its WebSocket service"
    ),
    "scripts/demos/heterogeneous_scene.py": ScriptOverride(
        args=("--num_task", "2"),
        readiness_pattern=r"Composed \d+ task scenes into \d+ environments",
    ),
    "scripts/demos/deformables.py": ScriptOverride(
        case_skip_reasons={
            ("isaacsim_physx", "default", "newton"): "Newton visualizer requires the Newton VBD physics backend",
            ("isaacsim_physx", "default", "rerun"): "Rerun cannot import PhysX deformable attributes",
            ("isaacsim_physx", "default", "viser"): "Viser cannot import PhysX deformable attributes",
        }
    ),
    "scripts/demos/mpm/newton_mpm_granular.py": ScriptOverride(
        args=("--max-steps", "20"),
        readiness_pattern=r"Newton granular MPM demo ready",
        fixed_physics_backend="newton_mpm",
    ),
    "scripts/demos/mpm/particle_pour.py": ScriptOverride(
        args=("--max-steps", "200"),
        readiness_pattern=r"particle-pour MPM demo ready",
        fixed_physics_backend="newton_mpm",
    ),
    "scripts/demos/multi_asset.py": ScriptOverride(args=("--num_envs", "4")),
    "scripts/demos/sensors/cameras.py": ScriptOverride(args=("--num_envs", "1"), startup_timeout=600.0),
    "scripts/demos/sensors/multi_mesh_raycaster.py": ScriptOverride(
        args=("--flat_ground",),
        startup_timeout=600.0,
        case_skip_reasons={
            ("newton_mjwarp", "default", "kit"): "Kit viewport fails with the Newton multi-mesh raycaster"
        },
    ),
    "scripts/demos/sensors/newton_raycast_heightfield.py": ScriptOverride(
        fixed_physics_backend="newton_mjwarp", visualizers=("none", "newton", "rerun", "viser")
    ),
    "scripts/demos/sensors/newton_raycast_moving_geometry.py": ScriptOverride(
        fixed_physics_backend="newton_mjwarp", visualizers=("none", "newton", "rerun", "viser")
    ),
    "scripts/demos/pick_and_place.py": ScriptOverride(
        readiness_pattern=r"Gym action space|Press the 'A' key", visualizers=("kit",)
    ),
    "scripts/demos/sensors/ppisp_camera.py": ScriptOverride(
        args=("--max_steps", "3", "--warmup_steps", "1", "--image_width", "64", "--image_height", "64"),
        startup_timeout=600.0,
        visualizers=("none",),
    ),
    "scripts/demos/sensors/ppisp_camera_ovrtx.py": ScriptOverride(
        args=("--max_steps", "3", "--warmup_steps", "1"),
        visualizers=("none",),
        required_modules=("ovrtx",),
    ),
    "scripts/tutorials/00_sim/create_empty.py": ScriptOverride(visualizers=("none", "kit")),
    "scripts/tutorials/00_sim/launch_app.py": ScriptOverride(visualizers=("none", "kit")),
    "scripts/tutorials/00_sim/log_time.py": ScriptOverride(visualizers=("none", "kit")),
    "scripts/tutorials/00_sim/spawn_prims.py": ScriptOverride(visualizers=("none", "kit")),
    "scripts/tutorials/01_assets/run_surface_gripper.py": ScriptOverride(
        args=("--device", "cpu"), visualizers=("none", "kit")
    ),
    "scripts/tutorials/03_envs/create_cartpole_base_env.py": ScriptOverride(readiness_pattern=r"Resetting environment"),
    "scripts/tutorials/03_envs/create_cube_base_env.py": ScriptOverride(readiness_pattern=r"Mean position error"),
    "scripts/tutorials/03_envs/create_quadruped_base_env.py": ScriptOverride(
        readiness_pattern=r"Resetting environment"
    ),
    "scripts/tutorials/03_envs/policy_inference_in_usd.py": ScriptOverride(
        skip_reason="requires a user-supplied TorchScript checkpoint"
    ),
    "scripts/tutorials/03_envs/run_cartpole_rl_env.py": ScriptOverride(readiness_pattern=r"Resetting environment"),
    "scripts/tutorials/04_sensors/add_sensors_on_robot.py": ScriptOverride(args=("--enable_cameras",)),
    "scripts/tutorials/04_sensors/run_ray_caster.py": ScriptOverride(visualizers=("none", "kit")),
    "scripts/tutorials/04_sensors/run_ray_caster_camera.py": ScriptOverride(
        args=("--enable_cameras",), visualizers=("none", "kit")
    ),
    "scripts/tutorials/04_sensors/run_usd_camera.py": ScriptOverride(
        args=("--enable_cameras",), visualizers=("none", "kit")
    ),
    "scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py": ScriptOverride(
        readiness_pattern=r"Gym action space",
        visualizers=("kit", "newton"),
    ),
}


def discover_specs() -> list[ScriptSpec]:
    """Discover executable demo/tutorial scripts and their literal CLI choices."""
    specs = []
    for root in SCRIPT_ROOTS:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            if not _has_main_guard(tree):
                continue
            relative_path = path.relative_to(ROOT).as_posix()
            override = OVERRIDES.get(relative_path, ScriptOverride())
            readiness_pattern = override.readiness_pattern
            if readiness_pattern is None and re.search(DEFAULT_READINESS_PATTERN, source):
                readiness_pattern = DEFAULT_READINESS_PATTERN
            options = _literal_cli_options(tree)
            visualizer_option = "--viz" if "--viz" in options and "--visualizer" not in options else "--visualizer"
            declared_visualizers = options.get(visualizer_option, ())
            if override.visualizers is not None:
                visualizers = override.visualizers
            elif declared_visualizers:
                visualizers = declared_visualizers
            else:
                visualizers = VISUALIZERS
            specs.append(
                ScriptSpec(
                    path=path,
                    options=options,
                    args=override.args,
                    readiness_pattern=readiness_pattern,
                    startup_timeout=override.startup_timeout,
                    skip_reason=override.skip_reason,
                    fixed_physics_backend=override.fixed_physics_backend,
                    visualizers=visualizers,
                    case_skip_reasons=override.case_skip_reasons,
                    visualizer_option=visualizer_option,
                    required_modules=override.required_modules,
                )
            )
    return specs


def build_cases(specs: list[ScriptSpec]) -> list[LaunchCase]:
    """Expand specs across declared physics, renderer, and visualizer choices."""
    return [
        LaunchCase(
            spec=spec,
            physics_option=physics_option,
            physics_backend=physics_backend,
            renderer_option=renderer_option,
            renderer_backend=renderer_backend,
            visualizer=visualizer,
        )
        for spec in specs
        for physics_option, physics_backend in spec.physics_backends
        for renderer_option, renderer_backend in spec.rendering_backends
        for visualizer in spec.visualizers
    ]


def select_script_scope(specs: list[ScriptSpec], scope: str) -> list[ScriptSpec]:
    """Select scripts within a repository-relative demo or tutorial directory.

    Args:
        specs: Discovered standalone script specifications.
        scope: Directory below ``scripts``, or ``"all"`` for every script.

    Returns:
        Specifications selected by the requested scope.

    Raises:
        ValueError: If a non-default scope does not select any scripts.
    """
    if scope == "all":
        return specs
    selected_specs = [spec for spec in specs if f"scripts/{scope}/" in spec.relative_path]
    if not selected_specs:
        raise ValueError(f"standalone script scope selected no scripts: {scope!r}")
    return selected_specs


def select_runtime_group(cases: list[LaunchCase], runtime_group: str) -> list[LaunchCase]:
    """Select launch cases by whether they require the Isaac Sim Kit runtime."""
    if runtime_group not in {"kit", "non-kit"}:
        raise ValueError(f"unsupported runtime group: {runtime_group!r}")
    return [
        case
        for case in cases
        if (
            case.physics_backend == "isaacsim_physx" or case.renderer_backend == "isaac_rtx" or case.visualizer == "kit"
        )
        == (runtime_group == "kit")
    ]


def backend_is_available(backend: str) -> bool:
    """Return whether the package implementing a selected backend is importable."""
    if backend == "default":
        return True
    if backend == "isaac_rtx":
        return importlib.util.find_spec("isaacsim") is not None or (ROOT / "_isaac_sim").exists()
    if backend in {"physx", "isaacsim_physx"}:
        package = "isaaclab_physx"
    else:
        package = "isaaclab_newton" if backend.startswith("newton") else f"isaaclab_{backend}"
    return importlib.util.find_spec(package) is not None


def module_is_available(module: str) -> bool:
    """Return whether an optional runtime module is importable."""
    return importlib.util.find_spec(module) is not None


def visualizer_is_available(visualizer: str) -> bool:
    """Return whether the package implementing a visualizer is importable."""
    if visualizer == "none":
        return True
    if visualizer == "kit":
        return importlib.util.find_spec("isaacsim") is not None or (ROOT / "_isaac_sim").exists()
    if importlib.util.find_spec("isaaclab_visualizers") is None:
        return False
    if visualizer == "newton":
        return importlib.util.find_spec("isaaclab_newton") is not None
    return importlib.util.find_spec(visualizer) is not None


def gui_is_available() -> bool:
    """Return whether the process has access to a desktop display server."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_until_ready(
    command: list[str],
    readiness_pattern: str,
    *,
    startup_timeout: float = 180.0,
    soak_time: float = 5.0,
    screenshot_path: Path | None = None,
    screenshot_delay: float = 1.0,
) -> SmokeResult:
    """Run a script until it exits or remains healthy after becoming ready.

    Infinite demos are terminated as a process group after the readiness marker
    has been observed and the soak interval has elapsed.
    """
    start_time = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = bytearray()
    ready_at = None
    stopped_after_soak = False
    screenshot_captured = False
    fatal_patterns = set()

    def record_output(chunk: bytes) -> None:
        """Record bounded output while retaining fatal-pattern state."""
        output.extend(chunk)
        decoded_output = output.decode(errors="replace")
        fatal_patterns.update(pattern for pattern in _FATAL_PATTERNS if pattern in decoded_output)
        if len(output) > MAX_OUTPUT_BYTES:
            del output[:-MAX_OUTPUT_BYTES]

    try:
        while True:
            for key, _ in selector.select(timeout=0.1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    record_output(chunk)
            decoded = output.decode(errors="replace")
            if ready_at is None and re.search(readiness_pattern, decoded):
                ready_at = time.monotonic()

            returncode = process.poll()
            if returncode is not None:
                break
            now = time.monotonic()
            if (
                screenshot_path is not None
                and ready_at is not None
                and not screenshot_captured
                and now - ready_at >= min(screenshot_delay, soak_time / 2.0)
            ):
                _capture_screenshot(screenshot_path)
                screenshot_captured = True
            if ready_at is not None and now - ready_at >= soak_time:
                stopped_after_soak = True
                _terminate_process_group(process)
                returncode = process.poll()
                break
            if now - start_time >= startup_timeout:
                _terminate_process_group(process)
                returncode = process.poll()
                break
    finally:
        selector.close()
        if process.poll() is None:
            _terminate_process_group(process)
        try:
            remainder, _ = process.communicate(timeout=5)
            record_output(remainder or b"")
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            remainder, _ = process.communicate(timeout=5)
            record_output(remainder or b"")

    decoded = output.decode(errors="replace")
    if ready_at is None and re.search(readiness_pattern, decoded):
        ready_at = time.monotonic()

    return SmokeResult(
        ready=ready_at is not None,
        returncode=process.returncode,
        output=decoded,
        elapsed=time.monotonic() - start_time,
        stopped_after_soak=stopped_after_soak,
        fatal_patterns=tuple(sorted(fatal_patterns)),
    )


def assert_smoke_passed(result: SmokeResult, case: LaunchCase) -> None:
    """Assert that a supervised script reached readiness without a fatal error."""
    tail = result.output[-30000:]
    assert not result.fatal_patterns, f"{case.id} emitted fatal output {result.fatal_patterns}:\n{tail}"
    assert result.ready, f"{case.id} did not reach {case.spec.readiness_pattern!r} in {result.elapsed:.1f}s:\n{tail}"
    assert result.stopped_after_soak or result.returncode == 0, (
        f"{case.id} exited with {result.returncode} before completing the soak:\n{tail}"
    )


def _has_main_guard(tree: ast.AST) -> bool:
    """Return whether an AST contains an executable ``__main__`` guard."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        names = [child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)]
        values = [child.value for child in ast.walk(node.test) if isinstance(child, ast.Constant)]
        if "__name__" in names and "__main__" in values:
            return True
    return False


def _literal_cli_options(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Collect literal ``argparse.add_argument`` options and choices from an AST."""
    options = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        option = node.args[0]
        if (
            not isinstance(option, ast.Constant)
            or not isinstance(option.value, str)
            or not option.value.startswith("--")
        ):
            continue
        choices: tuple[str, ...] = ()
        for keyword in node.keywords:
            if keyword.arg != "choices" or not isinstance(keyword.value, (ast.List, ast.Tuple)):
                continue
            literal_choices = [element.value for element in keyword.value.elts if isinstance(element, ast.Constant)]
            choices = tuple(str(choice) for choice in literal_choices)
        options[option.value] = choices
    return options


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate a script and any simulator children it spawned."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _capture_screenshot(path: Path) -> None:
    """Capture the active display to ``path`` using ImageMagick."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["import", "-window", "root", str(path)], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"failed to capture {path}: {result.stderr.strip()}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"screenshot command did not create a non-empty image at {path}")
