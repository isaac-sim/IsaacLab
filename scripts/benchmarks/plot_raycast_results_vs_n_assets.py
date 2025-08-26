# %% Imports
import io
import pandas as pd
import numpy as np
import plotly.express as px



#     mode  num_envs  resolution  rays_per_sensor  total_rays  per_step_ms   avg_memory  num_faces  track_mesh_transforms  num_assets
# 0  multi       512        0.05            10201     5222912   137.730139  5663.484688        320                   True           1
# 1  multi       512        0.05            10201     5222912   140.218571  5593.000937        320                   True           2
# 2  multi       512        0.05            10201     5222912   143.128282  5605.910937        320                   True           4
# 3  multi       512        0.05            10201     5222912   151.720909  5628.521875        320                   True           8
# 4  multi       512        0.05            10201     5222912   156.406283  5703.038750        320                   True          16
# 5  multi       512        0.05            10201     5222912   174.441749  6006.768437        320                   True          32
# 6  multi       512        0.05            10201     5222912   216.468570  6978.423125        320                   True          64
# 7  multi       512        0.05            10201     5222912   300.454327  8257.314688        320                   True         128


# %% Raw CSV (paste your data here) – OR replace with: df = pd.read_csv("your_file.csv")
df = pd.read_csv("/workspace/isaaclab/outputs/benchmarks/ray_caster_benchmark_vs_num_assets.csv")


# %% Types & cleaning
df["resolution"] = df["resolution"].astype(float)
df["num_envs"] = df["num_envs"].astype(int)
df["rays_per_sensor"] = df["rays_per_sensor"].astype(int)
df["total_rays"] = df["total_rays"].astype(int)
df["per_step_ms"] = df["per_step_ms"].astype(float)
df["avg_memory"] = df["avg_memory"].astype(float)
df["num_faces"] = df["num_faces"].replace(-1, np.nan)
df["num_assets"] = df["num_assets"].replace(-1, np.nan)
df["track_mesh_transforms"] = df["track_mesh_transforms"].replace(-1, np.nan)

# %% Derived metrics
df["rays_per_env"]  = df["total_rays"] / df["num_envs"]
df["time_per_ray_us"] = df["per_step_ms"] * 1000.0 / df["total_rays"]  # µs / ray
df["time_per_env_ms"] = df["per_step_ms"] / df["num_envs"]
df["mem_per_env"]   = df["avg_memory"] / df["num_envs"]

# ...existing code...

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors

# Calculate derived metrics
df["fps"] = 1.0 / (df["per_step_ms"]*1e-3)
df["rays_per_us"] = 1.0 / df["time_per_ray_us"]

# Create subplots: 3 rows, 1 column
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=[
        "FPS vs Number of Assets",
        "Average Memory vs Number of Assets",
        "Step Time (ms) vs Number of Assets"
    ],
    shared_xaxes=False
)

# Get unique resolution and num_envs for title
unique_res = ", ".join([str(r) for r in sorted(df["resolution"].unique())])
unique_envs = ", ".join([str(e) for e in sorted(df["num_envs"].unique())])

color_palette = plotly.colors.qualitative.Plotly
color_map = {res: color_palette[i % len(color_palette)] for i, res in enumerate(sorted(df["resolution"].unique()))}

# 1. FPS vs Number of Assets
for res in sorted(df["resolution"].unique()):
    filtered = df[df["resolution"] == res]
    fig.add_trace(
        go.Scatter(
            x=filtered["num_assets"],
            y=filtered["fps"],
            mode="lines+markers",
            name=f"Resolution: {res}",
            line=dict(color=color_map[res])
        ),
        row=1, col=1
    )

# 2. Average Memory vs Number of Assets
for res in sorted(df["resolution"].unique()):
    filtered = df[df["resolution"] == res]
    fig.add_trace(
        go.Scatter(
            x=filtered["num_assets"],
            y=filtered["avg_memory"],
            mode="lines+markers",
            name=f"Resolution: {res}",
            line=dict(color=color_map[res])
        ),
        row=2, col=1
    )

# 3. Step Time (ms) vs Number of Assets
for res in sorted(df["resolution"].unique()):
    filtered = df[df["resolution"] == res]
    fig.add_trace(
        go.Scatter(
            x=filtered["num_assets"],
            y=filtered["per_step_ms"],
            mode="lines+markers",
            name=f"Resolution: {res}",
            line=dict(color=color_map[res])
        ),
        row=3, col=1
    )

fig.update_xaxes(title_text="Number of Assets", row=1, col=1)
fig.update_yaxes(title_text="FPS", row=1, col=1)
fig.update_xaxes(title_text="Number of Assets", row=2, col=1)
fig.update_yaxes(title_text="Average Memory (MB)", row=2, col=1)
fig.update_xaxes(title_text="Number of Assets", row=3, col=1)
fig.update_yaxes(title_text="Step Time (ms)", row=3, col=1)

fig.update_layout(
    height=1200,
    title_text=f"Raycast Benchmark: Num Envs = {unique_envs}, Resolution = {unique_res}",
    legend_title_text="Resolution"
)

fig.show()

#