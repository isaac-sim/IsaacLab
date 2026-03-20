# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import cProfile
import glob
import os
import statistics
import sys

from tensorboard.backend.event_processing import event_accumulator

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, DictMeasurement, ListMeasurement, SingleMeasurement

# Path to configs.yaml and the config loader.
_BENCHMARKING_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "isaaclab_tasks", "test", "benchmarking"
)
_CONFIGS_YAML = os.path.join(_BENCHMARKING_DIR, "configs.yaml")


def get_backend_type(cli_backend: str) -> str:
    """Map old CLI backend names to new backend types.

    Args:
        cli_backend: The backend name from CLI arguments.

    Returns:
        The new backend type string.
    """
    mapping = {
        "OmniPerfKPIFile": "omniperf",
        "JSONFileMetrics": "json",
        "OsmoKPIFile": "osmo",
        "LocalLogMetrics": "json",
        "omniperf": "omniperf",
        "json": "json",
        "osmo": "osmo",
        "summary": "summary",
    }
    return mapping.get(cli_backend, "omniperf")


def parse_tf_logs(log_dir: str):
    """Search for the latest tfevents file in log_dir folder and returns
    the tensorboard logs in a dictionary.

    Args:
        log_dir: directory used to search for tfevents files
    """

    # search log directory for latest log file
    list_of_files = glob.glob(f"{log_dir}/events*")  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    log_data = {}
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in tags:
        log_data[tag] = []
        for event in ea.Scalars(tag):
            log_data[tag].append(event.value)

    return log_data


#############################
# logging benchmark metrics #
#############################


def log_min_max_mean_stats(benchmark: BaseIsaacLabBenchmark, values: dict):
    for k, v in values.items():
        unit = "FPS" if "FPS" in k else "ms" if "Time" in k or "time" in k else ""
        measurement = SingleMeasurement(name=f"Min {k}", value=min(v), unit=unit)
        benchmark.add_measurement("runtime", measurement=measurement)
        measurement = SingleMeasurement(name=f"Max {k}", value=max(v), unit=unit)
        benchmark.add_measurement("runtime", measurement=measurement)
        measurement = SingleMeasurement(name=f"Mean {k}", value=sum(v) / len(v), unit=unit)
        benchmark.add_measurement("runtime", measurement=measurement)


def log_app_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="App Launch Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_python_imports_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Python Imports Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_task_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Task Creation and Start Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_scene_creation_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Scene Creation Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_simulation_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Simulation Start Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_total_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Total Start Time (Launch to Train)", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_runtime_step_times(benchmark: BaseIsaacLabBenchmark, value: dict, compute_stats=True):
    measurement = DictMeasurement(name="Step Frametimes", value=value)
    benchmark.add_measurement("runtime", measurement=measurement)
    if compute_stats:
        log_min_max_mean_stats(benchmark, value)


def get_preset_string(hydra_args: list[str]) -> str:
    """Extract the active preset string from CLI hydra args or an environment variable.

    Checks (in order):
        1. ``presets=...`` in *hydra_args* (e.g. ``presets=physx,ovrtx_renderer,rgb``)
        2. ``ISAACLAB_BENCHMARK_PRESET`` environment variable
        3. Falls back to ``"default"``
    """
    for arg in hydra_args:
        if arg.startswith("presets="):
            value = arg.split("=", 1)[1]
            return value if value else "default"
    return os.environ.get("ISAACLAB_BENCHMARK_PRESET", "") or "default"


def log_rl_policy_rewards(benchmark: BaseIsaacLabBenchmark, value: list):
    measurement = ListMeasurement(name="Rewards", value=value)
    benchmark.add_measurement("train", measurement=measurement)
    # log max reward
    measurement = SingleMeasurement(name="Max Rewards", value=max(value), unit="float")
    benchmark.add_measurement("train", measurement=measurement)


def log_rl_policy_episode_lengths(benchmark: BaseIsaacLabBenchmark, value: list):
    measurement = ListMeasurement(name="Episode Lengths", value=value)
    benchmark.add_measurement("train", measurement=measurement)
    # log max episode length
    measurement = SingleMeasurement(name="Max Episode Lengths", value=max(value), unit="float")
    benchmark.add_measurement("train", measurement=measurement)


def check_convergence(
    rewards: list[float],
    threshold: float,
    window_pct: float = 0.2,
    cv_threshold: float = 20.0,
) -> dict:
    """Check whether training rewards have converged.

    Passes when the trailing window mean exceeds *threshold* and the
    coefficient of variation (CV) is below *cv_threshold*.

    Args:
        rewards: Per-iteration mean reward values.
        threshold: Minimum reward to pass.
        window_pct: Fraction of iterations for the trailing window.
        cv_threshold: Maximum CV (%) for stable convergence.

    Returns:
        Dict with ``tail_mean``, ``cv``, and ``passed``.
    """
    if not rewards:
        return {"tail_mean": 0.0, "cv": 999.9, "passed": False}
    window = max(1, int(len(rewards) * window_pct))
    tail = rewards[-window:]
    tail_mean = statistics.mean(tail)
    tail_std = statistics.stdev(tail) if len(tail) > 1 else 0.0
    cv = (tail_std / abs(tail_mean) * 100) if tail_mean != 0 else 999.9
    passed = tail_mean >= threshold and cv <= cv_threshold
    return {"tail_mean": round(tail_mean, 2), "cv": round(cv, 1), "passed": passed}


def log_convergence(
    benchmark: BaseIsaacLabBenchmark,
    rewards: list[float],
    task: str,
    workflow: str = "",
    should_check_convergence: bool = False,
    reward_threshold: float | None = None,
    convergence_config: str = "full",
):
    """Check reward convergence and log results to the benchmark backend.

    No-op unless *check_convergence* is True. When enabled, the threshold
    is loaded from ``configs.yaml``. *reward_threshold* overrides the config.

    Args:
        benchmark: Benchmark instance to log measurements to.
        rewards: Per-iteration mean reward values.
        task: Task name for config lookup.
        workflow: RL workflow name (``rsl_rl``, ``rl_games``, etc.).
        should_check_convergence: Whether ``--check_convergence`` was passed.
        reward_threshold: Explicit threshold override.
        convergence_config: Config section for threshold lookup (default: ``full``).
    """
    if not should_check_convergence:
        return

    threshold = reward_threshold
    if threshold is None and os.path.exists(_CONFIGS_YAML):
        if _BENCHMARKING_DIR not in sys.path:
            sys.path.insert(0, _BENCHMARKING_DIR)
        try:
            from env_benchmark_test_utils import get_env_config, get_env_configs

            entry = get_env_config(get_env_configs(_CONFIGS_YAML), convergence_config, workflow, task)
        except (ImportError, ValueError):
            entry = None
        if entry:
            threshold = entry.get("lower_thresholds", {}).get("reward")

    if threshold is None:
        print(
            f"[WARNING] No reward threshold found for '{task}'"
            f" in configs.yaml [{convergence_config}]. Skipping convergence check."
        )
        return

    result = check_convergence(rewards, threshold)
    benchmark.add_measurement(
        "train", SingleMeasurement(name="Mean Reward (Converged)", value=result["tail_mean"], unit="float")
    )
    benchmark.add_measurement("train", SingleMeasurement(name="Reward CV %", value=result["cv"], unit="%"))
    benchmark.add_measurement(
        "train", SingleMeasurement(name="Convergence Passed", value=int(result["passed"]), unit="bool")
    )


def parse_cprofile_stats(
    profile: cProfile.Profile,
    isaaclab_prefixes: list[str],
    top_n: int = 30,
    whitelist: list[str] | None = None,
) -> list[tuple[str, float, float]]:
    """Parse cProfile stats, filtering to IsaacLab + first-level external calls.

    Walks the pstats data and keeps functions that are either (a) inside an
    IsaacLab source directory, or (b) directly called by an IsaacLab function.
    Results are sorted by own-time (tottime) descending.

    When *whitelist* is provided, only functions whose labels match at least one
    ``fnmatch`` pattern are returned. Patterns that match no profiled function
    emit a ``(pattern, 0.0, 0.0)`` placeholder so dashboards always receive
    consistent keys. The *top_n* parameter is ignored in whitelist mode.

    Args:
        profile: A completed cProfile.Profile instance (after .disable()).
        isaaclab_prefixes: Absolute file path prefixes identifying IsaacLab source
            (e.g. ["/home/user/IsaacLab/source/isaaclab", ...]).
        top_n: Maximum number of functions to return. Ignored when
            *whitelist* is provided.
        whitelist: Optional list of ``fnmatch`` patterns to select specific
            functions (e.g. ``["isaaclab.cloner.*:usd_replicate"]``).

    Returns:
        List of (function_label, tottime_ms, cumtime_ms) tuples sorted by
        tottime descending.
    """
    import fnmatch
    import io
    import pstats

    stats = pstats.Stats(profile, stream=io.StringIO())

    def _is_isaaclab(filename: str) -> bool:
        return any(filename.startswith(prefix) for prefix in isaaclab_prefixes)

    def _make_label(filename: str, funcname: str) -> str:
        # For builtins/C-extensions the filename is something like "~" or "<frozen ...>"
        if not filename or filename.startswith("<") or filename == "~":
            return funcname
        # Convert absolute path to dotted module-style label
        for prefix in isaaclab_prefixes:
            if filename.startswith(prefix):
                rel = os.path.relpath(filename, prefix)
                # Strip .py, replace os.sep with dot
                rel = rel.replace(os.sep, ".").removesuffix(".py")
                return f"{rel}:{funcname}"
        # External function — try to find the top-level package name
        # e.g. ".../site-packages/torch/nn/modules/linear.py" -> "torch.nn.modules.linear"
        parts = filename.replace(os.sep, "/").removesuffix(".py").split("/")
        # Find "site-packages" anchor or fall back to last 3 components
        try:
            sp_idx = parts.index("site-packages")
            short = ".".join(parts[sp_idx + 1 :])
        except ValueError:
            short = ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts)
        return f"{short}:{funcname}"

    # stats.stats: dict[(filename, lineno, funcname)] -> (pcalls, ncalls, tottime, cumtime, callers)
    # callers: dict[(filename, lineno, funcname)] -> (pcalls, ncalls, tottime, cumtime)
    results = []
    for func_key, (_, _, tottime, cumtime, callers) in stats.stats.items():
        filename, _, funcname = func_key
        if _is_isaaclab(filename):
            label = _make_label(filename, funcname)
            results.append((label, tottime * 1000.0, cumtime * 1000.0))
        else:
            # Check if any direct caller is an IsaacLab function
            for caller_key in callers:
                caller_filename = caller_key[0]
                if _is_isaaclab(caller_filename):
                    label = _make_label(filename, funcname)
                    results.append((label, tottime * 1000.0, cumtime * 1000.0))
                    break

    # Sort by tottime (own-time) descending
    results.sort(key=lambda x: x[1], reverse=True)

    if whitelist is None:
        return results[:top_n]

    # Whitelist mode: filter by fnmatch patterns, emit placeholders for unmatched patterns
    matched: dict[str, tuple[str, float, float]] = {}
    matched_patterns: set[str] = set()
    for label, tottime, cumtime in results:
        for pattern in whitelist:
            if fnmatch.fnmatch(label, pattern):
                if label not in matched:
                    matched[label] = (label, tottime, cumtime)
                matched_patterns.add(pattern)

    # Add 0.0 placeholders for patterns that matched nothing
    for pattern in whitelist:
        if pattern not in matched_patterns:
            print(
                f"[WARNING] Whitelist pattern '{pattern}' matched no profiled functions. "
                "Check for typos or verify the function ran during this phase."
            )
            matched[pattern] = (pattern, 0.0, 0.0)

    filtered = list(matched.values())
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
