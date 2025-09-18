# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import pandas as pd
import plotly.express as px

import wandb


def compare_single_vs_multi():
    csv_path = "outputs/benchmarks/ray_caster_benchmark_single_vs_multi.csv"
    df = pd.read_csv(csv_path)

    artifact = wandb.Artifact("ray_caster_benchmark_single_vs_multi", type="dataset")
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)

    df["resolution"] = df["resolution"].astype(float)
    df["num_envs"] = df["num_envs"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["time_per_ray_us"] = df["per_step_ms"] * 1000.0 / df["total_rays"]
    df["rays_per_env"] = df["total_rays"] / df["num_envs"]
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)
    df["rays_per_us"] = 1.0 / df["time_per_ray_us"]

    df["resolution_label"] = df.apply(lambda r: f"{r['mode']} - {(5.0 / r['resolution'])**2:.2f} rays", axis=1)

    fig_fps = px.line(
        df,
        x="num_envs",
        y="fps",
        color="resolution_label",
        markers=True,
        title="FPS vs Num Envs",
    )

    fig_rays = px.line(
        df,
        x="num_envs",
        y="rays_per_us",
        color="resolution_label",
        markers=True,
        title="Rays per µs vs Num Envs",
    )

    fig_mem = px.line(
        df,
        x="num_envs",
        y="avg_memory",
        color="resolution_label",
        markers=True,
        title="Average VRAM Usage (MB) vs Num Envs",
    )

    wandb.log({
        "FPS vs Num Envs": fig_fps,
        "Rays per µs vs Num Envs": fig_rays,
        "VRAM vs Num Envs": fig_mem,
        "Single vs Multi Table": wandb.Table(dataframe=df),
    })


def compare_num_assets_vram_cache():
    csv_path = "outputs/benchmarks/ray_caster_benchmark_num_assets_reference.csv"
    df = pd.read_csv(csv_path)

    artifact = wandb.Artifact("ray_caster_benchmark_num_assets_reference", type="dataset")
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)

    df["num_assets"] = df["num_assets"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)
    df["reference_meshes"] = df["reference_meshes"].astype(bool)

    fig = px.line(
        df,
        x="num_assets",
        y="avg_memory",
        color="reference_meshes",
        markers=True,
        title="Average VRAM Usage (MB) vs Num Assets",
    )

    wandb.log({
        "Num Assets (VRAM Cache)": fig,
        "Num Assets VRAM Table": wandb.Table(dataframe=df),
    })


def compare_num_assets():
    csv_path = "outputs/benchmarks/ray_caster_benchmark_num_assets.csv"
    df = pd.read_csv(csv_path)

    artifact = wandb.Artifact("ray_caster_benchmark_num_assets", type="dataset")
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)

    df["num_assets"] = df["num_assets"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)

    fig_fps = px.line(
        df,
        x="num_assets",
        y="fps",
        markers=True,
        title="FPS vs Num Assets",
    )

    fig_mem = px.line(
        df,
        x="num_assets",
        y="avg_memory",
        markers=True,
        title="Average VRAM Usage (MB) vs Num Assets",
    )

    wandb.log({
        "FPS vs Num Assets": fig_fps,
        "VRAM vs Num Assets": fig_mem,
        "Num Assets Table": wandb.Table(dataframe=df),
    })


def compare_num_faces():
    csv_path = "outputs/benchmarks/ray_caster_benchmark_num_faces.csv"
    df = pd.read_csv(csv_path)

    artifact = wandb.Artifact("ray_caster_benchmark_num_faces", type="dataset")
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)

    df["num_faces"] = df["num_faces"].astype(int)
    df["avg_memory"] = df["avg_memory"].astype(float)
    df["fps"] = 1.0 / (df["per_step_ms"] * 1e-3)

    fig_fps = px.line(
        df,
        x="num_faces",
        y="fps",
        markers=True,
        title="FPS vs Num Faces",
    )

    fig_mem = px.line(
        df,
        x="num_faces",
        y="avg_memory",
        markers=True,
        title="Average VRAM Usage (MB) vs Num Faces",
    )

    wandb.log({
        "FPS vs Num Faces": fig_fps,
        "VRAM vs Num Faces": fig_mem,
        "Num Faces Table": wandb.Table(dataframe=df),
    })


if __name__ == "__main__":
    wandb.init(project="raycast-benchmarks", job_type="analysis", entity=os.getenv("WANDB_ENTITY", None))
    compare_single_vs_multi()
    compare_num_assets_vram_cache()
    compare_num_assets()
    compare_num_faces()
    wandb.finish()
