# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Command to run:
# uv run --no-sync python scripts/benchmarks/benchmark_renderer.py [PROFILE]
#

import argparse
import fnmatch
import json
import math
import os
import shutil
import site
import statistics
import subprocess
import sys
from pathlib import Path

OVRTX_RENDERER = "ovrtx_renderer"
NEWTON_RENDERER = "newton_renderer"

PROFILES = [
    {
        "name": f"ovrtx_{shading}_{pipeline}",
        "preset": f"{OVRTX_RENDERER},simple_shading_{shading}",
        "settings": {"min-pipe": minimal},
    }
    for shading in ("constant_diffuse", "diffuse_mdl", "full_mdl")
    for pipeline, minimal in (("oldpipe", False), ("newpipe", True))
] + [
    {
        "name": f"newton_{tlas}_{blas}",
        "preset": f"{NEWTON_RENDERER},rgb",
        "settings": {"tlas": tlas, "blas": blas},
    }
    for tlas in ("lbvh", "sah")
    for blas in ("lbvh", "sah", "cubql")
]

TASK_NAME = "Isaac-RenderBenchmark-Franka-Cabinet"
FRAME_PADDING = 5
STAT_KEYS = ("median", "mean", "min", "max", "stdev")

# Resolved from this file rather than the working directory, so the script runs from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_SCRIPT = SCRIPT_DIR / "runtime.py"
OUTPUT_PATH = str(SCRIPT_DIR.parent.parent / "benchmarks")

RENDER_SCOPE = "IsaacLab::Renderer::render"
"""Timer around ``BaseRenderer.render``, excluding scene updates and output readback."""

PHYSICS_SCOPE = "IsaacLab::Physics::step"
"""Timer around one physics step, matching :data:`isaaclab.benchmark.stepping.PHYSICS_PROFILE_SCOPE`."""

FRAME_SCOPE = "render"
"""Key of the :data:`PROFILE_SCOPES` entry whose timing closes a frame."""

PROFILE_SCOPES = {
    FRAME_SCOPE: RENDER_SCOPE,
    "physics": PHYSICS_SCOPE,
}
"""Report scope names mapped to timer names; :data:`FRAME_SCOPE` closes each frame."""

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
        The unqualified ``*_ms`` keys are the render ones; every other scope is prefixed with its
        name. ``pixels_per_second`` stays derived from the render time alone, so it remains
        comparable against a run whose physics cost differed.
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
    for group, stats in results.items():
        if isinstance(stats, dict):
            record |= {f"{group}_{key}_ms": stats[key] for key in STAT_KEYS}
    return record


TABLE_COLUMNS = [
    ("RENDER", "median_ms"),
    ("MEAN", "mean_ms"),
    ("MIN", "min_ms"),
    ("MAX", "max_ms"),
    ("STDEV", "stdev_ms"),
    ("PHYSICS", "physics_median_ms"),
    ("TOTAL", "total_median_ms"),
]
"""``(heading, record key)`` pairs for the report's timing columns, in display order."""


def format_table(records: list[dict]) -> list[str]:
    """Render the results table as lines of text.

    Widths follow the longest profile name rather than a fixed column, so a row stays aligned
    whichever profiles were selected.

    Args:
        records: Records from :func:`build_record`, in the order they should appear.

    Returns:
        The heading, separators, and one row per record.
    """
    name_width = max([len("PROFILE")] + [len(record["name"]) for record in records])
    separator = (
        "|" + "-" * (name_width + 2) + "|------|--------------|" + "|".join(["-" * 14] * len(TABLE_COLUMNS)) + "|"
    )
    headings = "|".join(f"{heading:^14}" for heading, _ in TABLE_COLUMNS)

    lines = ["| " + "PROFILE".ljust(name_width) + " | SIZE |  PIXEL / SEC |" + headings + "|", separator]
    for record in records:
        if record["status"] == "ok":
            cells = "|".join(f" {record[key]:>10.2f}ms " for _, key in TABLE_COLUMNS)
            gpxs = record["pixels_per_second"] / 1e9
            lines.append(f"| {record['name']:<{name_width}} | {record['size']:>4} | {gpxs:>6.2f} Gpx/s |{cells}|")
        else:
            lines.append(f"| {record['name']:<{name_width}} | FAILED {record['log']} |")
    lines.append(separator)
    return lines


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


def parse_frames(filename: str, scopes: dict[str, str] | None = None) -> list[dict[str, float]]:
    """Read per-frame scope times [ms] from a run's structured profiling file, in order.

    A rendered frame is preceded by however many physics steps the task's decimation implies, so
    non-frame scopes are accumulated until the frame scope's timing closes them out rather than
    assumed to be one per frame. A scope absent from the file reads as zero for every frame.

    Args:
        filename: Path to the profiling JSON file written by the runtime benchmark.
        scopes: Report scope name to timer name, defaulting to :data:`PROFILE_SCOPES`. Must contain
            :data:`FRAME_SCOPE`.

    Returns:
        One ``{scope: time_ms}`` dict per frame, each carrying every key in ``scopes``.

    Raises:
        TypeError: A timing entry has an invalid type.
        ValueError: The profiling file does not contain valid ordered scope timings.
    """
    scopes = PROFILE_SCOPES if scopes is None else scopes
    frames: list[dict[str, float]] = []
    pending = dict.fromkeys(scopes, 0.0)

    with open(filename) as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("timings_ms"), list):
        raise ValueError("Expected a 'timings_ms' list of [scope, elapsed_ms] pairs.")

    scope_names = {timer: name for name, timer in scopes.items()}
    for timer, elapsed_ms in payload["timings_ms"]:
        if type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("Expected a finite nonnegative time [ms].")
        if (name := scope_names.get(timer)) is None:
            continue
        if name == FRAME_SCOPE:
            frames.append(pending | {name: elapsed_ms})
            pending = dict.fromkeys(scopes, 0.0)
        else:
            pending[name] += elapsed_ms

    return frames


def parse_profile(filename: str, num_frames: int, scopes: dict[str, str] | None = None) -> dict | None:
    """Summarize per-frame times [ms] from a profiling JSON file, one entry per scope.

    Every backend is measured the same way: the wall time of :data:`RENDER_SCOPE`, collected once per
    render by ``wp.ScopedTimer`` when ``ISAACLAB_RENDER_PROFILE`` is set. The timer synchronizes the
    device on both ends, so it covers completed rather than merely submitted work — including for
    the RTX backends, whose Vulkan render is consumed by warp extraction kernels inside the scope.

    ``ISAACLAB_PHYSICS_PROFILE`` times :data:`PHYSICS_SCOPE` the same way, so a frame also carries
    what its physics steps cost and the two summed.

    Args:
        filename: Path to the profiling JSON file written by the runtime benchmark.
        num_frames: Number of frames to measure, after skipping :data:`FRAME_PADDING` warm-up frames.
        scopes: Report scope name to timer name, defaulting to :data:`PROFILE_SCOPES`.

    Returns:
        The :data:`FRAME_SCOPE` statistics flat, a sub-dict per remaining scope, and a ``total``
        sub-dict summing all of them. ``None`` if the file holds no usable frames.

    Raises:
        OSError: The profiling file cannot be read.
        TypeError: A timing entry has an invalid type.
        ValueError: The profiling file does not contain valid ordered scope timings.
    """
    scopes = PROFILE_SCOPES if scopes is None else scopes
    frames = parse_frames(filename, scopes)

    if not frames:
        log(f"No '{RENDER_SCOPE}' timings in {filename}; was ISAACLAB_RENDER_PROFILE set for this run?")
        return None
    out = frames[FRAME_PADDING : FRAME_PADDING + num_frames]

    if not out:
        return None

    results = {"size": len(out)} | summarize([frame[FRAME_SCOPE] for frame in out])
    results |= {name: summarize([frame[name] for frame in out]) for name in scopes if name != FRAME_SCOPE}
    return results | {"total": summarize([sum(frame.values()) for frame in out])}


def run_profile(profile: dict, args: argparse.Namespace):
    """Run one entry of :data:`PROFILES` and summarize its structured profiling results.

    Args:
        profile: Profile entry naming the preset and its backend settings.
        args: Parsed command-line arguments.

    Returns:
        Timing statistics from :func:`parse_profile`, or a false value if the run failed.
    """
    warp_cache_path = os.path.join(OUTPUT_PATH, "warp-cache")

    env = {
        "NEWTON_USE_CUDA_GRAPH": "0",
        "ISAACLAB_RENDER_PROFILE": "1",
        "ISAACLAB_PHYSICS_PROFILE": "1",
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

    output_path = os.path.join(OUTPUT_PATH, profile_name)
    os.makedirs(output_path, exist_ok=True)
    log_filename = os.path.join(OUTPUT_PATH, profile_name + ".log")
    profile_filename = os.path.join(output_path, "profile_timings.json")
    # A successful subprocess must produce its own measurements, never reuse a previous run's.
    Path(profile_filename).unlink(missing_ok=True)

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
        output_path,
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

    return parse_profile(profile_filename, args.num_frames)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, kept separate from module import so tests can import pure helpers."""
    parser = argparse.ArgumentParser("IsaacLab Benchmark: Sweep Franka Cabinet")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to render")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to render")
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution")
    parser.add_argument("--task", default=TASK_NAME, help="Gym task id to profile")
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
        sys.exit(0)

    matched_names = set()
    for profile_name in args.profile:
        matches = [profile["name"] for profile in PROFILES if fnmatch.fnmatch(profile["name"], profile_name)]
        if not matches:
            print(f"No profile found matching: {profile_name}", file=sys.stderr)
            sys.exit(1)
        matched_names.update(matches)

    # Run in declaration order so a given selection always reports in the same order.
    selected_profiles = [profile for profile in PROFILES if profile["name"] in matched_names]

    records = []
    for profile in selected_profiles:
        log(f"profile: {profile['name']}")
        log(f"  preset: {profile['preset']}")
        for key, value in profile["settings"].items():
            log(f"  {key}: {value}")

        try:
            results = run_profile(profile, args)
        except KeyboardInterrupt:
            break

        record = build_record(profile, results, args.num_envs, args.resolution)
        records.append(record)
        if record["status"] == "ok":
            log(f"    size: {record['size']}")
            for key, value in record.items():
                if key.endswith("_ms"):
                    log(f"    {key.removesuffix('_ms')}: {value:.2f}ms")
        log("")

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
        log("")
        for line in format_table(records):
            log(line)
        log("")

    sys.exit(1 if any(record["status"] == "failed" for record in records) else 0)


if __name__ == "__main__":
    main()
