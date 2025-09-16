# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt

import pandas as pd


def compare_single_vs_multi():
    df = pd.read_csv("outputs/benchmarks/ray_caster_benchmark_single_vs_multi.csv")
    # %% Types & cleaning
    df["resolution"] = df["resolution"].astype(float)
    df["num_envs"] = df["num_envs"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["time_per_ray_us"] = df["per_step_ms"] * 1000.0 / df["total_rays"]  # µs / ray
    df["rays_per_env"] = df["total_rays"] / df["num_envs"]
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)
    df["rays_per_us"] = 1.0 / df["time_per_ray_us"]

    unique_res = sorted(df["resolution"].unique())
    color_palette = plt.cm.tab10.colors
    color_map = {res: color_palette[i % len(color_palette)] for i, res in enumerate(unique_res)}
    dash_map = {"single": "-", "multi": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # 1. Steps per s vs Num Envs
    for res in unique_res:
        for mode in df["mode"].unique():
            sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
            axes[0].plot(
                sub_df["num_envs"],
                sub_df["fps"],
                dash_map.get(mode, "-"),
                color=color_map[res],
                marker="o",
                label=f"{mode} - Resolution={res}, Rays/Env={sub_df['rays_per_env'].values[0]}",
            )
    axes[0].set_xlabel("Number of Environments")
    axes[0].set_ylabel("FPS (frames per second)")
    axes[0].set_title("FPS vs Num Envs")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Rays per µs vs Num Envs
    for res in unique_res:
        for mode in df["mode"].unique():
            sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
            axes[1].plot(
                sub_df["num_envs"],
                sub_df["rays_per_us"],
                dash_map.get(mode, "-"),
                color=color_map[res],
                marker="o",
                label=f"{mode} - Resolution={res}, Rays/Env={sub_df['rays_per_env'].values[0]}",
            )
    axes[1].set_xlabel("Number of Environments")
    axes[1].set_ylabel("Rays per µs")
    axes[1].set_title("Rays per µs vs Num Envs")
    axes[1].legend()
    axes[1].grid(True)

    # 3. VRAM usage vs Number of Envs
    for res in unique_res:
        for mode in df["mode"].unique():
            sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
            axes[2].plot(
                sub_df["num_envs"],
                sub_df["avg_memory"],
                dash_map.get(mode, "-"),
                color=color_map[res],
                marker="o",
                label=f"{mode} - Resolution={res}, Rays/Env={sub_df['rays_per_env'].values[0]}",
            )
    axes[2].set_xlabel("Number of Environments")
    axes[2].set_ylabel("Average VRAM Usage (MB)")
    axes[2].set_title("Average VRAM Usage (MB) vs Num Envs")
    axes[2].legend()
    axes[2].grid(True)
    fig.suptitle("Raycast Benchmark Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save as PNG
    plt.savefig("outputs/benchmarks/raycast_benchmark_comparison.png")
    print("Saved plot to outputs/benchmarks/raycast_benchmark_comparison.png")


def compare_num_assets_vram_cache():
    """Plots Average steps per ms vs Num assets and Avg memory vs num assets"""
    df = pd.read_csv("outputs/benchmarks/ray_caster_benchmark_num_assets_reference.csv")
    df["num_assets"] = df["num_assets"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)
    df["reference_meshes"] = df["reference_meshes"].astype(bool)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    # 2. VRAM usage vs Number of Assets
    for ref in [True, False]:
        sub_df = df[df["reference_meshes"] == ref]
        axes.plot(sub_df["num_assets"], sub_df["avg_memory"], marker="o", label=f"Reference Meshes: {ref}")
    axes.set_xlabel("Number of Assets")
    axes.set_ylabel("Average VRAM Usage (MB)")
    axes.set_title("Average VRAM Usage (MB) vs Num Assets")
    axes.legend()
    axes.grid(True)

    fig.suptitle("Raycast Benchmark Comparison (Num Assets)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save as PNG
    plt.savefig("outputs/benchmarks/raycast_benchmark_comparison_num_assets_vram.png")
    print("Saved plot to outputs/benchmarks/raycast_benchmark_comparison_num_assets_vram.png")


def compare_num_assets():
    """Plots Average steps per ms vs Num assets and Avg memory vs num assets"""
    df = pd.read_csv("outputs/benchmarks/ray_caster_benchmark_num_assets.csv")
    df["num_assets"] = df["num_assets"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    # Plot FPS vs Num assets
    axes[0].plot(df["num_assets"], df["fps"], marker="o")
    axes[0].set_xlabel("Number of Assets")
    axes[0].set_ylabel("FPS")
    axes[0].set_title("FPS vs Num Assets")
    axes[0].grid(True)

    # 2. VRAM usage vs Number of Assets
    axes[1].plot(df["num_assets"], df["avg_memory"], marker="o")
    axes[1].set_xlabel("Number of Assets")
    axes[1].set_ylabel("Average VRAM Usage (MB)")
    axes[1].set_title("Average VRAM Usage (MB) vs Num Assets")
    axes[1].grid(True)

    fig.suptitle("Raycast Benchmark Comparison (Num Assets)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save as PNG
    plt.savefig("outputs/benchmarks/raycast_benchmark_comparison_num_assets.png")
    print("Saved plot to outputs/benchmarks/raycast_benchmark_comparison_num_assets.png")


def compare_num_faces():
    """Plots Average steps per ms vs Num faces and Avg memory vs num faces"""
    df = pd.read_csv("outputs/benchmarks/ray_caster_benchmark_num_faces.csv")
    # %% Types & cleaning
    df["num_faces"] = df["num_faces"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)

    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    # 1. Steps per ms vs Num Faces
    axes[0].plot(df["num_faces"], df["fps"], color="blue", marker="o", label="FPS")
    axes[0].set_xlabel("Number of Faces")
    axes[0].set_ylabel("FPS (frames per second)")
    axes[0].set_title("FPS vs Num Faces")
    axes[0].legend()
    axes[0].grid(True)

    # 2. VRAM usage vs Number of Faces
    axes[1].plot(df["num_faces"], df["avg_memory"], color="red", marker="o", label="Average VRAM Usage")
    axes[1].set_xlabel("Number of Faces")
    axes[1].set_ylabel("Average VRAM Usage (MB)")
    axes[1].set_title("Average VRAM Usage (MB) vs Num Faces")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle("Raycast Benchmark Comparison (Num Faces)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save as PNG
    plt.savefig("outputs/benchmarks/raycast_benchmark_comparison_num_faces.png")
    print("Saved plot to outputs/benchmarks/raycast_benchmark_comparison_num_faces.png")


if __name__ == "__main__":
    compare_single_vs_multi()
    compare_num_assets_vram_cache()
    compare_num_assets()
    compare_num_faces()
