# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
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
}
# Optional nice plotting style; fall back if not available
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science"]) 
except Exception:
    # Proceed with default matplotlib style if scienceplots isn't installed
    pass

SMALL_SIZE = 16
MEDIUM_SIZE = 20
LEGEND_SIZE = 15
BIGGER_SIZE = 24

plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", linewidth=2)

plt.figure(figsize=(6.0, 5.0), dpi=1200)

OUTPUT_DIR = "scripts/artifacts/"


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
    root_dir = os.path.dirname(os.path.normpath(OUTPUT_DIR))  # e.g., scripts/artifacts
    if any(ch in pattern_or_name for ch in ["*", "?"]):
        pattern = os.path.join(root_dir, "**", pattern_or_name)
    else:
        # Search for exact filename under all subfolders
        pattern = os.path.join(root_dir, "**", pattern_or_name)

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

            if "fps" in fields:
                df["fps"] = 1.0 / (df["per_step_ms"]) * df["num_envs"]

            if fields:
                keep = [c for c in fields if c in df.columns] + [c for c in keys if c in df.columns]
                if keep:
                    df = df[keep]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"Failed to load any CSVs for: {pattern_or_name}")

    df_all = pd.concat(frames, ignore_index=True)



    # Determine metrics = numeric fields that are in `fields` but not in `keys`
    keys_present = [k for k in keys if k in df_all.columns]
    metrics = []
    for c in fields:
        if c in keys_present:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df_all[c]):
                metrics.append(c)
        except Exception:
            continue
    if not metrics:
        return df_all

    agg_dict = {m: ["mean", "std"] for m in metrics}
    grouped = df_all.groupby(keys_present, dropna=False).agg(agg_dict).reset_index()
    # Flatten columns: keep mean under base name, std under base_name_std
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            flat_cols.append(base if stat == "mean" or stat == "" else f"{base}_{stat}")
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols
    # map device names to shorter versions if applicable
    if "device" in grouped.columns:
        grouped["device"] = grouped["device"].map(lambda x: device_name_lookup.get(x, x)).fillna(grouped["device"])
    return grouped


# plot1: FPS vs Num Assets.
def plot_num_assets_reference():
    df = get_dataframe(
        "num_assets_reference",
    fields=["avg_memory", "per_step_ms", "fps", "total_rays"],
        keys=["reference_meshes", "num_assets", "device", "num_envs", "resolution"],
    )
    df = df[df["reference_meshes"] == True]

    # df["reference_meshes"] = df["reference_meshes"].astype(bool)
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    for (device, ref), group in df.groupby(["device", "reference_meshes"]):
        label = f"{device}"
        # shade the std deviation
        axes.fill_between(
            group["num_assets"],
            group["fps"] - group["fps_std"],
            group["fps"] + group["fps_std"],
            alpha=0.2,
        )
        axes.plot(
            group["num_assets"],
            group["fps"],
            marker="o",
            label=label,
        )
    # Add right-hand tick labels as Rays/s using a conversion from FPS
    if set(["total_rays", "num_envs"]).issubset(df.columns) and len(df["num_envs"].unique()) == 1:
        rays_per_env = (df["total_rays"] / df["num_envs"]).median()
        if pd.notnull(rays_per_env) and rays_per_env > 0:
            fps_to_rps = lambda y: y * rays_per_env / 1e3
            rps_to_fps = lambda y: y / rays_per_env * 1e3
            ax_right = axes.secondary_yaxis('right', functions=(fps_to_rps, rps_to_fps))
            ax_right.set_ylabel("Throughput (Rays/s) $ \cdot 10^6$")
    axes.set_xlabel("Number of Assets")
    axes.set_ylabel("Throughput (FPS) $ \cdot 10^3$")
    axes.set_title(f"Ray Casting Performance vs Number of Assets for {df['num_envs'].iloc[0]} Envs")
    # axes.grid(True, which="both", linestyle="--", linewidth=0.5)
    axes.legend()
    plt.tight_layout()
    return axes

# plot2: FPS vs Mesh Complexity.
def plot_mesh_complexity():
    df = get_dataframe(
        "num_faces",
        fields=["avg_memory", "per_step_ms", "fps", "total_rays"],
        keys=["num_faces", "device", "num_envs", "resolution"],
    )

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    for device, group in df.groupby(["device"]):
        device = device[0]
        label = f"{device}"
        # shade the std deviation
        axes.fill_between(
            group["num_faces"],
            group["fps"] - group["fps_std"],
            group["fps"] + group["fps_std"],
            alpha=0.2,
        )
        axes.plot(
            group["num_faces"],
            group["fps"],
            marker="o",
            label=label,
        )
    # Add right-hand tick labels as Rays/s using a conversion from FPS
    if set(["total_rays", "num_envs"]).issubset(df.columns) and len(df["num_envs"].unique()) == 1:
        rays_per_env = (df["total_rays"] / df["num_envs"]).median()
        if pd.notnull(rays_per_env) and rays_per_env > 0:
            fps_to_rps = lambda y: y * rays_per_env / 1e3
            rps_to_fps = lambda y: y / rays_per_env * 1e3
            ax_right = axes.secondary_yaxis('right', functions=(fps_to_rps, rps_to_fps))
            ax_right.set_ylabel("Throughput (Rays/s) $ \cdot 10^6$")
    axes.set_xlabel("Mesh Complexity (Number of Faces)")
    axes.set_ylabel("Throughput (FPS) $ \cdot 10^3$")
    axes.set_title(f"Ray Casting Performance vs Mesh Complexity for {df['num_envs'].iloc[0]} Envs")
    # axes.grid(True, which="both", linestyle="--", linewidth=0.5)
    axes.legend()
    plt.tight_layout()
    return axes


def compare_single_vs_multi_resolution():
    df = get_dataframe(
        "single_vs_multi",
        fields=["avg_memory", "per_step_ms", "fps", "total_rays"],
        keys=["device", "num_envs", "resolution", "mode"],
    )
    df = df[df["mode"] == "multi"]
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    # build color and line-style maps: color per device, style per resolution
    devices = sorted(df["device"].unique())
    base_palette = list(plt.cm.tab10.colors)
    base_colors = {dev: base_palette[i % len(base_palette)] for i, dev in enumerate(devices)}
    style_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    style_map = {}
    for dev in devices:
        res_list = sorted(df[df["device"] == dev]["resolution"].unique())
        for idx, r in enumerate(res_list):
            style_map[(dev, r)] = style_cycle[idx % len(style_cycle)]

    # group by device and resolution
    for (device, res), group in df.groupby(["device", "resolution"]):
        label = f"{device} @ {(5.0 / res)**2:.1f} rays"
        # shade the std deviation with device color
        axes.fill_between(
            group["num_envs"],
            group["fps"] - group["fps_std"],
            group["fps"] + group["fps_std"],
            alpha=0.2,
            color=base_colors.get(device, None),
        )
        axes.plot(
            group["num_envs"],
            group["fps"],
            marker="o",
            label=label,
            linestyle=style_map.get((device, res), "-"),
            color=base_colors.get(device, None),
        )
    axes.set_xlabel("Number of Environments")
    axes.set_ylabel("Throughput (FPS) $ \cdot 10^3$")
    axes.set_title(f"Ray Casting Performance vs Number of Environments")
    axes.legend()
    return axes

def compare_memory_consumption():
    df = get_dataframe(
        "single_vs_multi",
        fields=["avg_memory", "per_step_ms", "fps", "total_rays"],
        keys=["device", "num_envs", "resolution", "mode"],
    )
    df = df[df["mode"] == "multi"]
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    # build color and line-style maps: color per device, style per resolution
    devices = sorted(df["device"].unique())
    base_palette = list(plt.cm.tab10.colors)
    base_colors = {dev: base_palette[i % len(base_palette)] for i, dev in enumerate(devices)}
    style_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    style_map = {}
    for dev in devices:
        res_list = sorted(df[df["device"] == dev]["resolution"].unique())
        for idx, r in enumerate(res_list):
            style_map[(dev, r)] = style_cycle[idx % len(style_cycle)]
    
    # zero_memory 
    min_memory = df["avg_memory"].min()
    # Normalize memory to start from zero
    df["avg_memory"] = df["avg_memory"] - min_memory 
    # group by device and resolution
    for (device, res), group in df.groupby(["device", "resolution"]):
        label = f"{device} @ {(5.0 / res)**2:.1f} rays"
        # shade the std deviation with device color
        axes.fill_between(
            group["num_envs"],
            group["avg_memory"] - group["avg_memory_std"],
            group["avg_memory"] + group["avg_memory_std"],
            alpha=0.2,
            color=base_colors.get(device, None),
        )
        axes.plot(
            group["num_envs"],
            group["avg_memory"],
            marker="o",
            label=label,
            linestyle=style_map.get((device, res), "-"),
            color=base_colors.get(device, None),
        )
    axes.set_xlabel("Number of Environments")
    axes.set_ylabel("Average Difference in VRAM Usage (MB)")
    axes.set_title(f"Ray Casting VRAM Usage vs Number of Environments")
    axes.legend()
    return axes

if __name__ == "__main__":
    axes = plot_num_assets_reference()
    # Save figure
    plt.savefig("num_assets_reference.png")
    axes = plot_mesh_complexity()
    plt.savefig("mesh_complexity.png")
    axes = compare_single_vs_multi_resolution()
    plt.savefig("single_vs_multi_resolution.png")
    axes = compare_memory_consumption()
    plt.savefig("memory_consumption.png")
    # Show all plots
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
