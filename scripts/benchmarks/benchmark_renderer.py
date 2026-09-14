# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Command to run:
# uv run --no-sync python scripts/benchmarks/benchmark_renderer.py [PROFILE]
#
# By default this measures the renderer alone. Pass --mode physics_render to drive the scene
# through the physics solver instead and report physics, render, and total times:
# uv run --no-sync python scripts/benchmarks/benchmark_renderer.py --mode physics_render [PROFILE]
#

import argparse
import fnmatch
import json
import os
import re
import shutil
import site
import statistics
import subprocess
import sys
from pathlib import Path

PROFILES = [
    {
        "name": "ovrtx_constant_diffuse_oldpipe",
        "preset": "ovrtx_renderer,simple_shading_constant_diffuse",
        "settings": {"min-pipe": False},
    },
    {
        "name": "ovrtx_constant_diffuse_newpipe",
        "preset": "ovrtx_renderer,simple_shading_constant_diffuse",
        "settings": {"min-pipe": True},
    },
    {
        "name": "ovrtx_diffuse_mdl_oldpipe",
        "preset": "ovrtx_renderer,simple_shading_diffuse_mdl",
        "settings": {"min-pipe": False},
    },
    {
        "name": "ovrtx_diffuse_mdl_newpipe",
        "preset": "ovrtx_renderer,simple_shading_diffuse_mdl",
        "settings": {"min-pipe": True},
    },
    {
        "name": "ovrtx_full_mdl_oldpipe",
        "preset": "ovrtx_renderer,simple_shading_full_mdl",
        "settings": {"min-pipe": False},
    },
    {
        "name": "ovrtx_full_mdl_newpipe",
        "preset": "ovrtx_renderer,simple_shading_full_mdl",
        "settings": {"min-pipe": True},
    },
    {"name": "newton_lbvh_lbvh", "preset": "newton_renderer,rgb", "settings": {"tlas": "lbvh", "blas": "lbvh"}},
    {"name": "newton_lbvh_sah", "preset": "newton_renderer,rgb", "settings": {"tlas": "lbvh", "blas": "sah"}},
    {"name": "newton_lbvh_cubql", "preset": "newton_renderer,rgb", "settings": {"tlas": "lbvh", "blas": "cubql"}},
    {"name": "newton_sah_lbvh", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "lbvh"}},
    {"name": "newton_sah_sah", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "sah"}},
    {"name": "newton_sah_cubql", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "cubql"}},
]

TASK_NAME = "Isaac-RenderBenchmark-Franka-Cabinet"
FRAME_PADDING = 5
STAT_KEYS = ("median", "mean", "min", "max", "stdev")

# Resolved from this file rather than the working directory, so the script runs from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_SCRIPT = SCRIPT_DIR / "runtime.py"
OUTPUT_PATH = str(SCRIPT_DIR.parent.parent / "benchmarks")

OVRTX_RENDERER = "ovrtx_renderer"
NEWTON_RENDERER = "newton_renderer"

RENDER_SCOPE = "IsaacLab::Renderer::render"
"""Backend-agnostic timer name around ``BaseRenderer.render``, enabled by ``ISAACLAB_RENDER_PROFILE``.

See :data:`isaaclab.renderers.render_context.RENDER_PROFILE_SCOPE`. It brackets the render alone,
excluding the scene-state sync before it and the output readback after it. ``wp.ScopedTimer`` prints
one ``"<name> took X.XX ms"`` line per call, which :func:`parse_log` regexes out of the run's log.
"""

PHYSICS_SCOPE = "IsaacLab::Physics::step"
"""Backend-agnostic timer name around one physics step, enabled by ``ISAACLAB_PHYSICS_PROFILE``.

See :data:`isaaclab.sim.simulation_context.PHYSICS_PROFILE_SCOPE`. Only emitted in
:data:`PHYSICS_RENDER_MODE`; a :data:`RENDER_MODE` run leaves the timer off so physics costs
nothing it would not cost in a normal run.
"""

RENDER_SCOPE_PATTERN = re.compile(rf"{re.escape(RENDER_SCOPE)} took ([\d.]+) ms")
PHYSICS_SCOPE_PATTERN = re.compile(rf"{re.escape(PHYSICS_SCOPE)} took ([\d.]+) ms")

RENDER_MODE = "render"
"""Benchmark mode that measures the renderer alone. See ``--mode``."""

PHYSICS_RENDER_MODE = "physics_render"
"""Benchmark mode that measures physics and the renderer together. See ``--mode``."""

MODES = (RENDER_MODE, PHYSICS_RENDER_MODE)
"""Values accepted by ``--mode``, mirroring the task's ``BenchmarkMode``."""

log_stream = sys.stdout
"""Destination for progress and diagnostics. ``--json`` points it at stderr so stdout holds only JSON."""


def log(message: str = "") -> None:
    """Write one human-readable line to :data:`log_stream`."""
    print(message, file=log_stream)


def pixels_per_second(median_ms: float, num_envs: int, resolution: int) -> float:
    """Rendered pixels per second [px/s] implied by a median frame time.

    Args:
        median_ms: Median time to render one frame [ms].
        num_envs: Number of environments rendered per frame.
        resolution: Width and height of each environment's tile [px].

    Returns:
        Throughput in pixels per second [px/s].
    """
    return num_envs * resolution * resolution * 1000.0 / median_ms


def summarize(samples: list[float]) -> dict:
    """Reduce a list of per-frame times [ms] to the statistics the report shows.

    Args:
        samples: One time per frame [ms]. Must be non-empty.

    Returns:
        Median, mean, min, max, and standard deviation [ms].
    """
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0,
    }


def build_record(profile: dict, results: dict | None, num_envs: int, resolution: int) -> dict:
    """Summarize one profile's outcome for reporting.

    Args:
        profile: Profile entry from :data:`PROFILES`.
        results: Timing statistics from :func:`parse_log`, or ``None`` if the run failed.
        num_envs: Number of environments the profile rendered.
        resolution: Width and height of each environment's tile [px].

    Returns:
        A record carrying the profile's identity plus either its timings or the log to inspect.
        ``physics_*_ms`` and ``total_*_ms`` are present only for a :data:`PHYSICS_RENDER_MODE` run;
        ``pixels_per_second`` is always derived from the render time alone, so it stays comparable
        between the two modes.
    """
    record = {"name": profile["name"], "preset": profile["preset"], "settings": profile["settings"]}
    if not results:
        return record | {"status": "failed", "log": os.path.join(OUTPUT_PATH, profile["name"] + ".log")}

    record |= {
        "status": "ok",
        "size": results["size"],
        "pixels_per_second": pixels_per_second(results["median"], num_envs, resolution),
    }
    record |= {f"{key}_ms": results[key] for key in STAT_KEYS}
    for group in ("physics", "total"):
        if results.get(group):
            record |= {f"{group}_{key}_ms": results[group][key] for key in STAT_KEYS}
    return record


def parse_frames(filename: str) -> list[tuple[float | None, float]]:
    """Read per-frame ``(physics_ms, render_ms)`` pairs out of a run's captured log, in order.

    A rendered frame is preceded by however many physics steps the task's decimation implies, so
    the physics timings are accumulated until the frame's render timing closes them out rather
    than assumed to be one per frame. ``physics_ms`` is ``None`` when the run did not enable
    physics profiling, which is every :data:`RENDER_MODE` run.

    Args:
        filename: Path to the captured run log.

    Returns:
        One ``(physics_ms, render_ms)`` pair per rendered frame.
    """
    frames: list[tuple[float | None, float]] = []
    physics_ms = 0.0
    saw_physics = False

    with open(filename) as file:
        for line in file:
            if match := PHYSICS_SCOPE_PATTERN.search(line):
                physics_ms += float(match.group(1))
                saw_physics = True
            elif match := RENDER_SCOPE_PATTERN.search(line):
                frames.append((physics_ms if saw_physics else None, float(match.group(1))))
                physics_ms = 0.0
                saw_physics = False

    return frames


def parse_log(filename: str, num_frames: int):
    """Summarize per-frame times [ms] from a run's captured log.

    Every backend is measured the same way: the wall time of :data:`RENDER_SCOPE`, printed once per
    render by ``wp.ScopedTimer`` when ``ISAACLAB_RENDER_PROFILE`` is set. The timer synchronizes the
    device on both ends, so it covers completed rather than merely submitted work — including for
    the RTX backends, whose Vulkan render is consumed by warp extraction kernels inside the scope.

    A :data:`PHYSICS_RENDER_MODE` run also sets ``ISAACLAB_PHYSICS_PROFILE``, which times
    :data:`PHYSICS_SCOPE` the same way; those frames additionally report the physics time and the
    two summed.

    Args:
        filename: Path to the captured run log.
        num_frames: Number of frames to measure, after skipping :data:`FRAME_PADDING` warm-up frames.

    Returns:
        Render timing statistics, plus ``physics`` and ``total`` sub-dicts when the log holds
        physics timings, or ``None`` if the log holds no usable frames.
    """
    frames = parse_frames(filename)

    if not frames:
        log(f"No '{RENDER_SCOPE}' timings in {filename}; was ISAACLAB_RENDER_PROFILE set for this run?")
        return None
    out = frames[FRAME_PADDING : FRAME_PADDING + num_frames]

    if not out:
        return None

    results = {"size": len(out)} | summarize([render_ms for _, render_ms in out])

    # Partial physics coverage would mix frames measured under different timers, so only report
    # physics when every measured frame carries it.
    if all(physics_ms is not None for physics_ms, _ in out):
        results["physics"] = summarize([physics_ms for physics_ms, _ in out])
        results["total"] = summarize([physics_ms + render_ms for physics_ms, render_ms in out])

    return results


def table_columns(records: list[dict]) -> list[tuple[str, str]]:
    """Choose the report's timing columns from what the records actually carry.

    A :data:`RENDER_MODE` sweep has no physics timings, so its table stays the render-only table
    it has always been; a :data:`PHYSICS_RENDER_MODE` sweep gains the physics and total medians.

    Args:
        records: Records from :func:`build_record`.

    Returns:
        ``(heading, record key)`` pairs for the timing columns, in display order.
    """
    columns = [("MEDIAN", "median_ms"), ("MEAN", "mean_ms"), ("MIN", "min_ms"), ("MAX", "max_ms")]
    if any("physics_median_ms" in record for record in records):
        columns += [("PHYSICS", "physics_median_ms"), ("TOTAL", "total_median_ms")]
    return columns + [("STDEV", "stdev_ms")]


def format_table(records: list[dict]) -> list[str]:
    """Render the results table as lines of text.

    Args:
        records: Records from :func:`build_record`, in the order they should appear.

    Returns:
        The heading, separators, and one row per record.
    """
    columns = table_columns(records)
    name_width = max([len("PROFILE")] + [len(record["name"]) for record in records])
    separator = "|" + "-" * (name_width + 2) + "|------|--------------|" + "|".join(["-" * 14] * len(columns)) + "|"

    lines = [
        "| "
        + "PROFILE".ljust(name_width)
        + " | SIZE |  PIXEL / SEC |"
        + "|".join(f"{h:^14}" for h, _ in columns)
        + "|",
        separator,
    ]
    for record in records:
        if record["status"] == "ok":
            # A column can be absent for an individual record (a profile whose run emitted no
            # physics timings alongside profiles that did); leave that cell blank rather than
            # dropping the column for everyone.
            cells = "|".join(f" {record[key]:>10.2f}ms " if key in record else " " * 14 for _, key in columns)
            gpxs = record["pixels_per_second"] / 1e9
            lines.append(f"| {record['name']:<{name_width}} | {record['size']:>4} | {gpxs:>6.2f} Gpx/s |{cells}|")
        else:
            lines.append(f"| {record['name']:<{name_width}} | FAILED {record['log']} |")
    lines.append(separator)
    return lines


def run_profile(profile: dict, args: argparse.Namespace):
    """Run one entry of :data:`PROFILES` and summarize its render times from the captured log.

    Args:
        profile: Profile entry naming the preset and its backend settings.
        args: Parsed command-line arguments.

    Returns:
        Timing statistics from :func:`parse_log`, or ``False`` if the run failed.
    """
    warp_cache_path = os.path.join(OUTPUT_PATH, "warp-cache")

    env = {
        "NEWTON_USE_CUDA_GRAPH": "0",
        "ISAACLAB_RENDER_PROFILE": "1",
        # Left off in render mode so the physics timer's device synchronization does not serialize
        # work around a step whose cost the run is deliberately not measuring.
        "ISAACLAB_PHYSICS_PROFILE": "1" if args.mode == PHYSICS_RENDER_MODE else "0",
        "BENCHMARK_MODE": args.mode,
        "BENCHMARK_SAVE_IMAGE": "1" if args.save_image else "0",
        "BENCHMARK_RENDER_RESOLUTION": f"{args.resolution}",
        "WARP_CACHE_PATH": warp_cache_path,
    }

    # Kernel compilation lands in the warm-up frames that FRAME_PADDING discards, so a warm cache
    # does not reach the measured window -- it only saves a recompile per profile.
    if not args.keep_warp_cache and os.path.exists(warp_cache_path):
        shutil.rmtree(warp_cache_path)

    profile_name: str = profile["name"]
    preset: str = profile["preset"]
    renderer: str = preset.split(",")[0]

    if renderer == OVRTX_RENDERER:
        env["LD_PRELOAD"] = os.path.join(
            site.getsitepackages()[0], "ovrtx/bin/plugins/omni.client.lib/libomniclient.so"
        )
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["OVRTX_rtx_post_tonemap_op"] = "0"
        env["OVRTX_rtx_minimal_useMinimalPipeline"] = "1" if profile["settings"]["min-pipe"] else "0"

    if renderer == NEWTON_RENDERER:
        env["NEWTON_BVH_SCENE"] = profile["settings"]["tlas"]
        env["NEWTON_BVH_GEOMETRY"] = profile["settings"]["blas"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    log_filename = os.path.join(OUTPUT_PATH, profile_name + ".log")

    cmd = [
        sys.executable,
        str(RUNTIME_SCRIPT),
        "--task",
        args.task,
        "--num_envs",
        f"{args.num_envs}",
        "--warmup_steps",
        "0",
        "--num_steps",
        f"{args.num_frames + FRAME_PADDING * 2}",
        "--output_path",
        OUTPUT_PATH,
        f"presets={preset}",
    ]

    with open(log_filename, "w") as file:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ) | env, text=True
        )

        for line in process.stdout:
            file.write(line)
            if args.verbose:
                log_stream.write(f"\x1b[90m{line}\x1b[0m")
                log_stream.flush()

        return_code = process.wait()
        if return_code != 0:
            log(f"Failed with exit code {process.returncode}, see {log_filename} for details.")
            return False

    return parse_log(log_filename, args.num_frames)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, kept separate from module import so tests can import pure helpers."""
    parser = argparse.ArgumentParser("IsaacLab Benchmark: Sweep Franka Cabinet")
    parser.add_argument("--num_frames", type=int, default="20", help="Number of frames to render")
    parser.add_argument("--num_envs", type=int, default="1024", help="Number of environments to render")
    parser.add_argument("--resolution", type=int, default="256", help="Render resolution")
    parser.add_argument("--task", default=TASK_NAME, help="Gym task id to profile")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=RENDER_MODE,
        help=(
            "What to measure: 'render' times the renderer alone on a directly posed scene;"
            " 'physics_render' drives the same poses through the actuators and reports physics,"
            " render, and total times"
        ),
    )
    parser.add_argument(
        "--keep_warp_cache",
        action="store_true",
        help="Reuse the warp kernel cache instead of recompiling per profile",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-s", "--save_image", action="store_true", help="Save image for debugging purposes")
    parser.add_argument("--json", action="store_true", help="Write results to stdout as JSON instead of a table")
    parser.add_argument("profile", nargs="*", help="Profiles to run")
    return parser


def main() -> None:
    """Parse CLI arguments, run the selected profiles, and report their results."""
    global log_stream

    args = _build_arg_parser().parse_args()

    # Keep stdout free of anything but the JSON document.
    if args.json:
        log_stream = sys.stderr

    if not args.profile:
        if args.json:
            print(json.dumps({"available_profiles": [profile["name"] for profile in PROFILES]}, indent=2))
        else:
            log("Available profiles:")
            for profile in PROFILES:
                log("  " + profile["name"])
        exit(0)

    matched_names = set()
    for profile_name in args.profile:
        matches = [profile["name"] for profile in PROFILES if fnmatch.fnmatch(profile["name"], profile_name)]
        if not matches:
            print(f"No profile found matching: {profile_name}", file=sys.stderr)
            exit(1)
        matched_names.update(matches)

    # Run in declaration order so a given selection always reports in the same order.
    selected_profiles = [profile for profile in PROFILES if profile["name"] in matched_names]

    all_results = {}
    for profile in selected_profiles:
        log(f"profile: {profile['name']}")
        log(f"  preset: {profile['preset']}")
        for key, value in profile["settings"].items():
            log(f"  {key}: {value}")

        try:
            all_results[profile["name"]] = run_profile(profile, args)
        except KeyboardInterrupt:
            break

        if results := all_results[profile["name"]]:
            log(f"    size: {results['size']}")
            for label, stats in (
                ("render", results),
                ("physics", results.get("physics")),
                ("total", results.get("total")),
            ):
                if stats:
                    log(f"    {label}: " + "  ".join(f"{key}={stats[key]:.2f}ms" for key in STAT_KEYS))
        log("")

    # A KeyboardInterrupt leaves the remaining profiles unrun; report only what completed.
    records = [
        build_record(profile, all_results[profile["name"]], args.num_envs, args.resolution)
        for profile in selected_profiles
        if profile["name"] in all_results
    ]
    benchmark_failed = any(record["status"] == "failed" for record in records)

    if args.json:
        print(
            json.dumps(
                {
                    "task": args.task,
                    "mode": args.mode,
                    "num_envs": args.num_envs,
                    "num_frames": args.num_frames,
                    "resolution": args.resolution,
                    "profiles": records,
                },
                indent=2,
            )
        )
    else:
        log("")
        for line in format_table(records):
            log(line)
        log("")

    if benchmark_failed:
        exit(1)
    exit(0)


if __name__ == "__main__":
    main()
