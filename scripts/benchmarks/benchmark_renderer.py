# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Command to run:
# uv run --no-sync python scripts/benchmarks/benchmark_renderer.py [PROFILE]
#

import argparse
import contextlib
import fnmatch
import json
import os
import shutil
import site
import sqlite3
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

# Resolved from this file rather than the working directory, so the script runs from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
NSYS_TRACE_CONFIG = SCRIPT_DIR / "nsys_trace.json"
RUNTIME_SCRIPT = SCRIPT_DIR / "runtime.py"
OUTPUT_PATH = str(SCRIPT_DIR.parent.parent / "benchmarks")

OVRTX_RENDERER = "ovrtx_renderer"
NEWTON_RENDERER = "newton_renderer"

RENDER_SCOPE = "IsaacLab::Renderer::render"
"""Backend-agnostic NVTX range around ``BaseRenderer.render``, enabled by ``ISAACLAB_RENDER_PROFILE``.

See :data:`isaaclab.renderers.render_context.RENDER_PROFILE_SCOPE`. It brackets the render alone,
excluding the scene-state sync before it and the output readback after it.
"""

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


def build_record(profile: dict, results: dict | None, num_envs: int, resolution: int) -> dict:
    """Summarize one profile's outcome for reporting.

    Args:
        profile: Profile entry from :data:`PROFILES`.
        results: Timing statistics from :func:`parse_profile`, or ``None`` if the run failed.
        num_envs: Number of environments the profile rendered.
        resolution: Width and height of each environment's tile [px].

    Returns:
        A record carrying the profile's identity plus either its timings or the log to inspect.
    """
    record = {"name": profile["name"], "preset": profile["preset"], "settings": profile["settings"]}
    if not results:
        return record | {"status": "failed", "log": os.path.join(OUTPUT_PATH, profile["name"] + ".log")}
    return (
        record
        | {
            "status": "ok",
            "size": results["size"],
            "pixels_per_second": pixels_per_second(results["median"], num_envs, resolution),
        }
        | {f"{key}_ms": results[key] for key in ("median", "mean", "min", "max", "stdev")}
    )


def nvtx_ranges(cursor: sqlite3.Connection, name: str) -> list[tuple[int, int]]:
    """Fetch every NVTX range labelled ``name``, ordered by start time.

    A range carries its label inline or through ``StringIds`` depending on whether the emitter
    registered the string, so both are matched.

    Args:
        cursor: Connection to an exported nsys SQLite report.
        name: Exact range label to match.

    Returns:
        ``(start, end)`` pairs in nanoseconds.
    """
    return cursor.execute(
        """
        SELECT n.start, n.end FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE COALESCE(s.value, n.text) = ?
        ORDER BY n.start
        """,
        (name,),
    ).fetchall()


def parse_profile(filename: str, num_frames: int):
    """Summarize per-frame render times [ms] from an exported nsys report.

    Every backend is measured the same way: the wall time of :data:`RENDER_SCOPE`. The range
    synchronizes the device on both ends, so it covers completed rather than merely submitted work
    — including for the RTX backends, whose Vulkan render is consumed by warp extraction kernels
    inside the range.

    Args:
        filename: Path to the nsys SQLite export.
        num_frames: Number of frames to measure, after skipping :data:`FRAME_PADDING` warm-up frames.

    Returns:
        Timing statistics, or ``None`` if the report holds no usable frames.
    """
    with contextlib.closing(sqlite3.connect(filename)) as cursor:
        frames = nvtx_ranges(cursor, RENDER_SCOPE)

    if not frames:
        log(f"No '{RENDER_SCOPE}' ranges in {filename}; was ISAACLAB_RENDER_PROFILE set for this run?")
        return None
    out = [
        (end - start) / 1e6
        for i, (start, end) in enumerate(frames)
        if FRAME_PADDING < i + 1 <= (num_frames + FRAME_PADDING)
    ]

    if out:
        return {
            "size": len(out),
            "median": statistics.median(out),
            "mean": statistics.mean(out),
            "min": min(out),
            "max": max(out),
            "stdev": statistics.stdev(out) if len(out) > 1 else 0,
        }
    return None


def run_profile(profile: dict, args: argparse.Namespace):
    """Profile one entry of :data:`PROFILES` under nsys and summarize its render times.

    Args:
        profile: Profile entry naming the preset and its backend settings.
        args: Parsed command-line arguments.

    Returns:
        Timing statistics from :func:`parse_profile`, or ``False`` if the run or the export failed.
    """
    warp_cache_path = os.path.join(OUTPUT_PATH, "warp-cache")

    env = {
        "NVTX_PROFILE_PYTHON": "1",
        "NVTX_PROFILE_INCLUDE": "isaaclab,newton,warp,rsl_rl",
        "NEWTON_USE_CUDA_GRAPH": "0",
        "ISAACLAB_RENDER_PROFILE": "1",
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
    trace: str = "nvtx"

    if renderer == OVRTX_RENDERER:
        env["LD_PRELOAD"] = os.path.join(
            site.getsitepackages()[0], "ovrtx/bin/plugins/omni.client.lib/libomniclient.so"
        )
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["OVRTX_rtx_post_tonemap_op"] = "0"
        env["OVRTX_rtx_minimal_useMinimalPipeline"] = "1" if profile["settings"]["min-pipe"] else "0"
        env["OVRTX_app_profilerBackend"] = "nvtx"
        env["OVRTX_app_profileFromStart"] = "true"
        env["OVRTX_app_profilerMask"] = "1"

    if renderer == NEWTON_RENDERER:
        trace = "nvtx,cuda"
        env["NEWTON_BVH_SCENE"] = profile["settings"]["tlas"]
        env["NEWTON_BVH_GEOMETRY"] = profile["settings"]["blas"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    profile_filename = os.path.join(OUTPUT_PATH, profile_name + ".nsys-rep")
    log_filename = os.path.join(OUTPUT_PATH, profile["name"] + ".log")

    cmd = [
        "nsys",
        "profile",
        "--output",
        profile_filename,
        "--force-overwrite",
        "true",
        "--trace",
        trace,
        "--python-functions-trace",
        str(NSYS_TRACE_CONFIG),
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

    # A failed export leaves the previous run's .sqlite in place, which would then be parsed and
    # reported as if it were this run's.
    export = subprocess.run(
        [
            "nsys",
            "stats",
            "--force-export",
            "true",
            "--report",
            "cuda_kern_exec_sum",
            "--format",
            "csv",
            profile_filename,
        ],
        stdout=subprocess.PIPE,
    )
    if export.returncode != 0:
        log(f"'nsys stats' failed with exit code {export.returncode} for {profile_filename}.")
        return False
    return parse_profile(profile_filename.replace(".nsys-rep", ".sqlite"), args.num_frames)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, kept separate from module import so tests can import pure helpers."""
    parser = argparse.ArgumentParser("IsaacLab Benchmark: Sweep Franka Cabinet")
    parser.add_argument("--num-frames", type=int, default="20", help="Number of frames to render")
    parser.add_argument("--num-envs", type=int, default="1024", help="Number of environments to render")
    parser.add_argument("--resolution", type=int, default="256", help="Render resolution")
    parser.add_argument("--task", default=TASK_NAME, help="Gym task id to profile")
    parser.add_argument(
        "--keep-warp-cache",
        action="store_true",
        help="Reuse the warp kernel cache instead of recompiling per profile",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-s", "--save-image", action="store_true", help="Save image for debugging purposes")
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
            for key, value in results.items():
                if key != "size":
                    log(f"    {key}: {value:.2f}ms")
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
                    "num_envs": args.num_envs,
                    "num_frames": args.num_frames,
                    "resolution": args.resolution,
                    "profiles": records,
                },
                indent=2,
            )
        )
    else:
        separator = "|------------------------------------------|------|--------------|--------------|--------------|--------------|--------------|--------------|"  # noqa: E501
        log("")
        log(
            "| PROFILE                                  | SIZE |  PIXEL / SEC |    MEDIAN    |     MEAN     |     MIN      |     MAX      |    STDEV     |"  # noqa: E501
        )
        log(separator)
        for record in records:
            if record["status"] == "ok":
                gpxs = record["pixels_per_second"] / 1e9
                log(
                    f"| {record['name']:<40} | {record['size']:>4} | {gpxs:>6.2f} Gpx/s | {record['median_ms']:>10.2f}ms | {record['mean_ms']:>10.2f}ms | {record['min_ms']:>10.2f}ms | {record['max_ms']:>10.2f}ms | {record['stdev_ms']:>10.2f}ms |"  # noqa: E501
                )
            else:
                log(f"| {record['name']:<40} | FAILED {record['log']:<87} |")
        log(separator)
        log("")

    if benchmark_failed:
        exit(1)
    exit(0)


if __name__ == "__main__":
    main()
