# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
import argparse
import os
import glob
import numpy as np

import pandas as pd

device_name_lookup = {
    "NVIDIA RTX 3060": "RTX 3060",
    "NVIDIA RTX 3090": "RTX 3090",
    "NVIDIA RTX 4090": "RTX 4090",
    "NVIDIA RTX 4080": "RTX 4080",
    "NVIDIA RTX 4070 Ti": "RTX 4070 Ti",
    "NVIDIA RTX 4060 Ti": "RTX 4060 Ti",
    "NVIDIA RTX 3080": "RTX 3080",
    "NVIDIA RTX 2080 Ti": "RTX 2080 Ti",
    "NVIDIA RTX 2080 SUPER": "RTX 2080 SUPER",
    "NVIDIA RTX 2080": "RTX 2080",
    "NVIDIA GTX 1080 Ti": "GTX 1080 Ti",
    "NVIDIA GTX 1080": "GTX 1080",
    "NVIDIA GTX 1070": "GTX 1070",
    "NVIDIA GTX 1060": "GTX 1060",
    "NVIDIA GTX 980 Ti": "GTX 980 Ti",
    "NVIDIA GTX 980": "GTX 980",
    "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "Blackwell 6000",
}


registered_devices = []
registered_resolutions = {}

def get_color(device: str, resolution: float) -> str:
    """Get a color string based on device and resolution."""
    cmap = cm.get_cmap('tab20c')
    base_palette = [cmap(i) for i in range(cmap.N)]
    
    if device not in registered_devices:
        registered_devices.append(device)
        registered_resolutions[device] = []
    if resolution is not None and resolution not in registered_resolutions[device]:
        registered_resolutions[device].append(resolution)
    
    base_color_idx = (registered_devices.index(device) * 4) % 20
    if resolution is None:
        fine_color_idx = base_color_idx
    else:
        fine_color_idx = registered_resolutions[device].index(resolution) % 4 + base_color_idx

    return base_palette[fine_color_idx % len(base_palette)]


import scienceplots  # noqa: F401
plt.style.use(["science"])  # publication-ready base style

SMALL_SIZE = 16
MEDIUM_SIZE = 20
LEGEND_SIZE = 15
BIGGER_SIZE = 24

plt.rc('font', size=14)  # base font size
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=LEGEND_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rc("lines", linewidth=2)

OUTPUT_DIR = os.environ.get("BENCHMARK_PLOTS_DIR", "scripts/outputs/nvidia/")


def _format_axes(ax: plt.Axes, grid: bool = True):
    """Apply consistent, publication-ready formatting to axes."""
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    # Use scalar formatter without offset/scientific on axes unless necessary
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def get_dataframe(df_name: str, fields=None, keys=None) -> pd.DataFrame:
    """Load CSVs (glob-aware) and optionally aggregate mean/std on requested fields.

    Accepted logical names:
      - "single_vs_multi" -> ray_caster_benchmark_single_vs_multi.csv
      - "num_assets_reference" -> ray_caster_benchmark_num_assets_reference.csv
      - "num_assets" -> ray_caster_benchmark_num_assets.csv
      - "num_faces" -> ray_caster_benchmark_num_faces.csv

    If df_name ends with .csv, it will look for that filename; if df_name contains
    wildcard characters (*, ?), it will be treated as a glob pattern. Matching
    is recursive under the parent folder of OUTPUT_DIR (e.g., scripts/artifacts/**).

                Parameters
                - df_name: logical name or CSV filename (or glob pattern)
                - fields: optional list of column names to include; if provided, only
                    existing columns from this list are kept. When None, all columns are kept.
                - keys: optional list of column names to group by for aggregation. When
                    provided (and multiple files match), numeric fields in `fields` that are
                    not group keys will be aggregated with mean/std. If not provided, data is
                    concatenated without aggregation.
    """
    mapping = {
        "single_vs_multi": "ray_caster_benchmark_single_vs_multi.csv",
        "num_assets_reference": "ray_caster_benchmark_num_assets_reference.csv",
        "num_assets": "ray_caster_benchmark_num_assets.csv",
        "num_faces": "ray_caster_benchmark_num_faces.csv",
    }


    pattern_or_name = mapping.get(df_name, df_name)
    # Determine search root and pattern
    root_dir = os.path.normpath(OUTPUT_DIR) # e.g., scripts/artifacts
    print(f"Searching for CSVs matching '{pattern_or_name}' under '{root_dir}'")
    if any(ch in pattern_or_name for ch in ["*", "?"]):
        pattern = os.path.join(root_dir, "**", pattern_or_name)
        print(f"Using glob pattern: {pattern}")
    else:
        # Search for exact filename under all subfolders
        pattern = os.path.join(root_dir, "**", pattern_or_name)
        print(f"Using exact filename search with pattern: {pattern}")

    matches = glob.glob(pattern, recursive=True)
    # Fallback: direct path inside OUTPUT_DIR
    if not matches:
        direct = os.path.join(OUTPUT_DIR, pattern_or_name)
        if os.path.exists(direct):
            matches = [direct]
    if not matches:
        raise FileNotFoundError(f"No CSV files found for pattern/name: {pattern_or_name} under {root_dir}")

    # Load and optionally column-filter
    frames = []
    print(f"Loading {len(matches)} CSV files for pattern/name: {pattern_or_name}")

    for p in matches:
        try:
            df = pd.read_csv(p)

            # Derived metrics
            if fields and ("fps" in fields or "kfps" in fields or "mrays_per_s" in fields):
                # Frames per second across all envs (aggregate FPS)
                df["fps"] = 1e3 / df["per_step_ms"] * df["num_envs"]  # total FPS
                df["kfps"] = df["fps"] / 1e3
                if "total_rays" in df.columns:
                    # Throughput in Mray/s
                    df["mrays_per_s"] = (df["fps"] * (df["total_rays"] / df["num_envs"])) / 1e6

            if fields:
                keep = [c for c in (fields or []) if c in df.columns] + [c for c in (keys or []) if c in df.columns]
                if keep:
                    df = df[keep]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"Failed to load any CSVs for: {pattern_or_name}")

    df_all = pd.concat(frames, ignore_index=True)



    # Determine metrics = numeric fields that are in `fields` but not in `keys`
    keys_present = [k for k in (keys or []) if k in df_all.columns]
    metrics: list[str] = []
    for c in (fields or []):
        if c in keys_present:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df_all[c]):
                metrics.append(c)
        except Exception:
            continue
    if not metrics or not keys_present:
        return df_all

    group = df_all.groupby(keys_present, dropna=False)
    mean_df = group[metrics].mean().reset_index()
    std_df = group[metrics].std().reset_index()
    # Rename std columns with _std suffix
    std_df = std_df.rename(columns={m: f"{m}_std" for m in metrics})
    grouped = pd.merge(mean_df, std_df, on=keys_present, how="left")
    # map device names to shorter versions if applicable
    if "device" in grouped.columns:
        # Map known device names to short labels; keep original where not mapped
        grouped["device"] = grouped["device"].map(device_name_lookup).fillna(grouped["device"])
    return grouped


# plot1: FPS vs Num Assets.
def plot_num_assets_reference():
    df = get_dataframe(
        "num_assets_reference",
        fields=["avg_memory", "per_step_ms", "fps", "kfps", "mrays_per_s", "total_rays"],
        keys=["reference_meshes", "num_assets", "device", "num_envs", "resolution"],
    )
    df = df[df["reference_meshes"] == True]

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.61), dpi=600, constrained_layout=True)
    for (device, ref), group in df.groupby(["device", "reference_meshes"]):
        group = group.sort_values("num_assets")
        color = get_color(device, None)
        label = f"{device}"
        # shade the std deviation
        x = group["num_assets"].to_numpy(dtype=float)
        y = (group["kfps"].to_numpy(dtype=float) if "kfps" in group.columns else (group["fps"].to_numpy(dtype=float) / 1e3))
        ystd = (
            group["kfps_std"].to_numpy(dtype=float) if "kfps_std" in group.columns else
            (group["fps_std"].to_numpy(dtype=float) / 1e3 if "fps_std" in group.columns else np.zeros_like(y))
        )
        axes.fill_between(x, y - ystd, y + ystd, alpha=0.2, color=color)
        axes.plot(x, y, marker="o", label=label, color=color)
    # Secondary axis as Mray/s
    if set(["total_rays", "num_envs"]).issubset(df.columns) and len(df["num_envs"].unique()) == 1:
        rays_per_env = (df["total_rays"] / df["num_envs"]).median()
        if pd.notnull(rays_per_env) and rays_per_env > 0:
            kfps_to_mrays = lambda y: y * rays_per_env / 1e3
            mrays_to_kfps = lambda y: y * 1e3 / rays_per_env
            ax_right = axes.secondary_yaxis('right', functions=(kfps_to_mrays, mrays_to_kfps))
            ax_right.set_ylabel(r"Throughput (Rays / s) $\times 10^6$")
    axes.set_xlabel("Number of assets")
    axes.set_ylabel(r"Throughput (FPS) $\times 10^3$")
    #axes.set_title(f"Throughput vs number of assets ({int(df['num_envs'].iloc[0])} envs)")
    _format_axes(axes)
    axes.legend(frameon=False)
    return axes

# plot2: FPS vs Mesh Complexity.
def plot_mesh_complexity():
    df = get_dataframe(
        "num_faces",
    fields=["avg_memory", "per_step_ms", "fps", "kfps", "mrays_per_s", "total_rays"],
        keys=["num_faces", "device", "num_envs", "resolution"],
    )

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.61), dpi=600, constrained_layout=True)
    for device, group in df.groupby(["device"]):
        group = group.sort_values("num_faces")
        color = get_color(device, None)
        label = f"{device[0]}"
        # shade the std deviation
        x = group["num_faces"].to_numpy(dtype=float)
        y = (group["kfps"].to_numpy(dtype=float) if "kfps" in group.columns else (group["fps"].to_numpy(dtype=float) / 1e3))
        ystd = (
            group["kfps_std"].to_numpy(dtype=float) if "kfps_std" in group.columns else
            (group["fps_std"].to_numpy(dtype=float) / 1e3 if "fps_std" in group.columns else np.zeros_like(y))
        )
        axes.fill_between(x, y - ystd, y + ystd, alpha=0.2, color=color)
        axes.plot(x, y, marker="o", label=label, color=color)
    # Add right-hand tick labels as Mray/s using a conversion from kFPS
    if set(["total_rays", "num_envs"]).issubset(df.columns) and len(df["num_envs"].unique()) == 1:
        rays_per_env = (df["total_rays"] / df["num_envs"]).median()
        if pd.notnull(rays_per_env) and rays_per_env > 0:
            kfps_to_mrays = lambda y: y * rays_per_env / 1e3
            mrays_to_kfps = lambda y: y * 1e3 / rays_per_env
            ax_right = axes.secondary_yaxis('right', functions=(kfps_to_mrays, mrays_to_kfps))
            ax_right.set_ylabel(r"Throughput (Rays) $\times 10^6$")
    axes.set_xlabel("Mesh complexity (faces)")
    axes.set_ylabel(r"Throughput (FPS) $\times 10^3$")
    #axes.set_title(f"Throughput vs mesh complexity ({int(df['num_envs'].iloc[0])} envs)")
    _format_axes(axes)
    axes.legend(frameon=False)
    return axes


def compare_single_vs_multi_resolution():
    df = get_dataframe(
        "single_vs_multi",
        fields=["avg_memory", "per_step_ms", "fps", "kfps", "mrays_per_s", "total_rays"],
        keys=["device", "num_envs", "resolution", "mode"],
    )
    # df = df[df["mode"] == "multi"]
    
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.61), dpi=600, constrained_layout=True)
    # build color and line-style maps: color per device, style per resolution
    devices = sorted(df["device"].unique())
    style_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    style_map = {}
    for dev in devices:
        res_list = sorted(df[df["device"] == dev]["resolution"].unique())
        for idx, r in enumerate(res_list):
            style_map[(dev, r)] = style_cycle[idx % len(style_cycle)]

    idx = 0
    # group by device and resolution
    for (device, res, mode), group in df.groupby(["device", "resolution", "mode"]):
        if mode != "multi":
            continue
        color = get_color(device, res)
        group = group.sort_values("num_envs")
        label = f"{device} - ({(5.0 / res)**1:.0f}$\\times${(5.0 / res)**1:.0f})" # - {mode}"
        # shade the std deviation with device color
        x = group["num_envs"].to_numpy(dtype=float)
        y = (group["mrays_per_s"].to_numpy(dtype=float) if "mrays_per_s" in group.columns else (group["fps"].to_numpy(dtype=float) / 1e3))
        ystd = (
            group["mrays_per_s_std"].to_numpy(dtype=float) if "mrays_per_s_std" in group.columns else
            (group["fps_std"].to_numpy(dtype=float) / 1e3 if "fps_std" in group.columns else np.zeros_like(y))
        )
        axes.fill_between(list(x), list(y - ystd), list(y + ystd), alpha=0.2, color=color)
        axes.plot(list(x), list(y), marker="o", label=label, linestyle=style_map.get((device, res), "-"), color=color)
    axes.set_xlabel("Number of environments")
    axes.set_ylabel(r"Throughput (Rays / s) $\times 10^6$")
    #axes.set_title("Throughput vs number of environments")
    _format_axes(axes)
    axes.legend(frameon=False, ncol=1, loc="best")
    return axes


def compare_single_vs_multi_fps():
    df = get_dataframe(
        "single_vs_multi",
        fields=["avg_memory", "per_step_ms", "fps", "kfps", "mrays_per_s", "total_rays"],
        keys=["device", "num_envs", "resolution", "mode"],
    )
    # df = df[df["mode"] == "multi"]
    
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.61), dpi=600, constrained_layout=True)
    # build color and line-style maps: color per device, style per resolution
    devices = sorted(df["device"].unique())
    style_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    style_map = {}
    for dev in devices:
        res_list = sorted(df[df["device"] == dev]["resolution"].unique())
        for idx, r in enumerate(res_list):
            style_map[(dev, r)] = style_cycle[idx % len(style_cycle)]

    idx = 0
    # group by device and resolution
    for (device, res, mode), group in df.groupby(["device", "resolution", "mode"]):
        if mode != "multi":
            continue
        color = get_color(device, res)
        group = group.sort_values("num_envs")
        label = f"{device} - ({(5.0 / res)**1:.0f}$\\times${(5.0 / res)**1:.0f})" # - {mode}"
        # shade the std deviation with device color
        x = group["num_envs"].to_numpy(dtype=float)
        y = (group["kfps"].to_numpy(dtype=float) if "kfps" in group.columns else (group["fps"].to_numpy(dtype=float) / 1e3))
        ystd = (
            group["kfps_std"].to_numpy(dtype=float) if "kfps_std" in group.columns else
            (group["fps_std"].to_numpy(dtype=float) / 1e3 if "fps_std" in group.columns else np.zeros_like(y))
        )
        axes.fill_between(list(x), list(y - ystd), list(y + ystd), alpha=0.2, color=color)
        axes.plot(list(x), list(y), marker="o", label=label, linestyle=style_map.get((device, res), "-"), color=color)
    axes.set_xlabel("Number of environments")
    axes.set_ylabel(r"Throughput (FPS) $\times 10^3$")
    #axes.set_title("Throughput vs number of environments")
    _format_axes(axes)
    axes.legend(frameon=False, ncol=1, loc="best")
    return axes

def compare_memory_consumption():
    df = get_dataframe(
        "single_vs_multi",
        fields=["avg_memory", "per_step_ms", "fps", "kfps", "mrays_per_s", "total_rays"],
        keys=["device", "num_envs", "resolution", "mode"],
    )
    df = df[df["mode"] == "multi"]
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.61), dpi=600, constrained_layout=True)
    # build color and line-style maps: color per device, style per resolution
    devices = sorted(df["device"].unique())
    style_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    style_map = {}
    for dev in devices:
        res_list = sorted(df[df["device"] == dev]["resolution"].unique())
        for idx, r in enumerate(res_list):
            style_map[(dev, r)] = style_cycle[idx % len(style_cycle)]
    
    # group by device and resolution
    for (device, res), group in df.groupby(["device", "resolution"]):
        label = f"{device} - ({(5.0 / res)**1:.0f}$\\times${(5.0 / res)**1:.0f}) rays"
        color = get_color(device, res)
        # shade the std deviation with device color
        x = group["num_envs"].to_numpy(dtype=float)
        # normalize each series by its own minimum
        y_raw = group["avg_memory"].to_numpy(dtype=float)
        # use nanmin to be safe if NaNs are present; fall back to 0 if all NaN
        y_min = np.nanmin(y_raw) if np.any(~np.isnan(y_raw)) else 0.0
        y = y_raw - y_min
        ystd = group["avg_memory_std"].to_numpy(dtype=float) if "avg_memory_std" in group.columns else np.zeros_like(y)
        axes.fill_between(x, y - ystd, y + ystd, alpha=0.2, color=color)
        axes.plot(x, y, marker="o", label=label, linestyle=style_map.get((device, res), "-"), color=color)
    axes.set_xlabel("Number of environments")
    axes.set_ylabel(r"VRAM usage $\Delta$ (MB)")
    # saxes.set_title("VRAM usage vs number of environments")
    _format_axes(axes)
    axes.legend(frameon=False, loc="best")
    return axes

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        axes = plot_num_assets_reference()
        plt.savefig(os.path.join(OUTPUT_DIR, "num_assets_reference.png"), dpi=800)
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'num_assets_reference.png')}")
    except Exception as e:
        print(f"Failed to plot num_assets_reference: {e}")

    try:
        axes = plot_mesh_complexity()
        plt.savefig(os.path.join(OUTPUT_DIR, "mesh_complexity.png"), dpi=800)
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'mesh_complexity.png')}")
    except Exception as e:
        print(f"Failed to plot mesh_complexity: {e}")
        
    try:
        axes = compare_single_vs_multi_resolution()
        plt.savefig(os.path.join(OUTPUT_DIR, "single_vs_multi_resolution.png"), dpi=800)
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'single_vs_multi_resolution.png')}")
    except Exception as e:
        print(f"Failed to plot single_vs_multi_resolution: {e}")
        
    try:
        axes = compare_single_vs_multi_fps()
        plt.savefig(os.path.join(OUTPUT_DIR, "single_vs_multi_fps.png"), dpi=800)
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'single_vs_multi_fps.png')}")
    except Exception as e:
        print(f"Failed to plot single_vs_multi_fps: {e}")
        
    try:
        axes = compare_memory_consumption()
        plt.savefig(os.path.join(OUTPUT_DIR, "memory_consumption.png"), dpi=800)
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'memory_consumption.png')}")
    except Exception as e:
        print(f"Failed to plot memory_consumption: {e}")
    # Show all plots (optional; comment out for headless)
    plt.show()


# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# import os

# import pandas as pd
# import plotly.express as px

# import wandb


# def compare_single_vs_multi():
#     csv_path = "outputs/benchmarks/ray_caster_benchmark_single_vs_multi.csv"
#     df = pd.read_csv(csv_path)

#     artifact = wandb.Artifact("ray_caster_benchmark_single_vs_multi", type="dataset")
#     artifact.add_file(csv_path)
#     wandb.log_artifact(artifact)

#     df["resolution"] = df["resolution"].astype(float)
#     df["num_envs"] = df["num_envs"].astype(int)
#     df["avg_memory"] = df["avg_memory"].astype(float)
#     df["time_per_ray_us"] = df["per_step_ms"] * 1000.0 / df["total_rays"]
#     df["rays_per_env"] = df["total_rays"] / df["num_envs"]
#     df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)
#     df["rays_per_us"] = 1.0 / df["time_per_ray_us"]

#     df["resolution_label"] = df.apply(lambda r: f"{r['mode']} - {(5.0 / r['resolution'])**2:.2f} rays", axis=1)

#     fig_fps = px.line(
#         df,
#         x="num_envs",
#         y="fps",
#         color="resolution_label",
#         markers=True,
#         title="FPS vs Num Envs",
#     )

#     fig_rays = px.line(
#         df,
#         x="num_envs",
#         y="rays_per_us",
#         color="resolution_label",
#         markers=True,
#         title="Rays per µs vs Num Envs",
#     )

#     fig_mem = px.line(
#         df,
#         x="num_envs",
#         y="avg_memory",
#         color="resolution_label",
#         markers=True,
#         title="Average VRAM Usage (MB) vs Num Envs",
#     )

#     wandb.log({
#         "FPS vs Num Envs": fig_fps,
#         "Rays per µs vs Num Envs": fig_rays,
#         "VRAM vs Num Envs": fig_mem,
#         "Single vs Multi Table": wandb.Table(dataframe=df),
#     })


# def compare_num_assets_vram_cache():
#     csv_path = "outputs/benchmarks/ray_caster_benchmark_num_assets_reference.csv"
#     df = pd.read_csv(csv_path)

#     artifact = wandb.Artifact("ray_caster_benchmark_num_assets_reference", type="dataset")
#     artifact.add_file(csv_path)
#     wandb.log_artifact(artifact)

#     df["num_assets"] = df["num_assets"].astype(int)
#     df["avg_memory"] = df["avg_memory"].astype(float)
#     df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)
#     df["reference_meshes"] = df["reference_meshes"].astype(bool)

#     fig = px.line(
#         df,
#         x="num_assets",
#         y="avg_memory",
#         color="reference_meshes",
#         markers=True,
#         title="Average VRAM Usage (MB) vs Num Assets",
#     )

#     wandb.log({
#         "Num Assets (VRAM Cache)": fig,
#         "Num Assets VRAM Table": wandb.Table(dataframe=df),
#     })


# def compare_num_assets():
#     csv_path = "outputs/benchmarks/ray_caster_benchmark_num_assets.csv"
#     df = pd.read_csv(csv_path)

#     artifact = wandb.Artifact("ray_caster_benchmark_num_assets", type="dataset")
#     artifact.add_file(csv_path)
#     wandb.log_artifact(artifact)

#     df["num_assets"] = df["num_assets"].astype(int)
#     df["avg_memory"] = df["avg_memory"].astype(float)
#     df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)

#     fig_fps = px.line(
#         df,
#         x="num_assets",
#         y="fps",
#         markers=True,
#         title="FPS vs Num Assets",
#     )

#     fig_mem = px.line(
#         df,
#         x="num_assets",
#         y="avg_memory",
#         markers=True,
#         title="Average VRAM Usage (MB) vs Num Assets",
#     )

#     wandb.log({
#         "FPS vs Num Assets": fig_fps,
#         "VRAM vs Num Assets": fig_mem,
#         "Num Assets Table": wandb.Table(dataframe=df),
#     })


# def compare_num_faces():
#     csv_path = "outputs/benchmarks/ray_caster_benchmark_num_faces.csv"
#     df = pd.read_csv(csv_path)

#     artifact = wandb.Artifact("ray_caster_benchmark_num_faces", type="dataset")
#     artifact.add_file(csv_path)
#     wandb.log_artifact(artifact)

#     df["num_faces"] = df["num_faces"].astype(int)
#     df["avg_memory"] = df["avg_memory"].astype(float)
#     df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)

#     fig_fps = px.line(
#         df,
#         x="num_faces",
#         y="fps",
#         markers=True,
#         title="FPS vs Num Faces",
#     )

#     fig_mem = px.line(
#         df,
#         x="num_faces",
#         y="avg_memory",
#         markers=True,
#         title="Average VRAM Usage (MB) vs Num Faces",
#     )

#     wandb.log({
#         "FPS vs Num Faces": fig_fps,
#         "VRAM vs Num Faces": fig_mem,
#         "Num Faces Table": wandb.Table(dataframe=df),
#     })


# if __name__ == "__main__":
#     wandb.init(project="raycast-benchmarks", job_type="analysis", entity=os.getenv("WANDB_ENTITY", None))
#     compare_single_vs_multi()
#     compare_num_assets_vram_cache()
#     compare_num_assets()
#     compare_num_faces()
#     wandb.finish()
