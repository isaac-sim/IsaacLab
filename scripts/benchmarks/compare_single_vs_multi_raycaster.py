# %% Imports
import io
import pandas as pd
import numpy as np
import plotly.express as px



#       mode  num_envs  resolution  rays_per_sensor  total_rays  per_step_ms   avg_memory  num_faces track_mesh_transforms  num_assets
# 0   single       256        0.20              676      173056    18.660888  4917.913125         -1                    -1          -1
# 3    multi       256        0.20              676      173056    14.664904  5217.812500         -1                  True           0

# 6   single       512        0.20              676      346112    38.912707  5198.900312         -1                    -1          -1
# 9    multi       512        0.20              676      346112    29.210159  5701.625000         -1                  True           0

# 1   single       256        0.10             2601      665856    41.373501  4977.825000         -1                    -1          -1
# 4    multi       256        0.10             2601      665856    32.879590  5176.748750         -1                  True           0

# 15   multi      1024        0.20              676      692224    60.821424  6690.753125         -1                  True           0
# 12  single      1024        0.20              676      692224    77.989069  5659.967813         -1                    -1          -1

# 7   single       512        0.10             2601     1331712    90.059855  5268.675000         -1                    -1          -1
# 10   multi       512        0.10             2601     1331712    64.682159  5669.915000         -1                  True           0

# 2   single       256        0.05            10201     2611456    74.297123  5217.812500         -1                    -1          -1
# 5    multi       256        0.05            10201     2611456    70.873075  5263.118438         -1                  True           0

# 16   multi      1024        0.10             2601     2663424   135.986122  6798.684063         -1                  True           0
# 13  single      1024        0.10             2601     2663424   165.702515  5642.069688         -1                    -1          -1

# 8   single       512        0.05            10201     5222912   151.106479  5720.017500         -1                    -1          -1
# 11   multi       512        0.05            10201     5222912   139.969725  5642.006562         -1                  True           0

# 14  single      1024        0.05            10201    10445824   315.265636  6626.664687         -1                    -1          -1
# 17   multi      1024        0.05            10201    10445824   282.527562  6876.511875         -1                  True           0


# %% Raw CSV (paste your data here) – OR replace with: df = pd.read_csv("your_file.csv")
df = pd.read_csv("/workspace/isaaclab/outputs/benchmarks/ray_caster_benchmark_single_vs_multi.csv")


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



# Plot Comparison. x-axis: num_envs, y axis 1/per_step_ms for all unique resolution levels.
df["fps"] = 1.0 / (df["per_step_ms"]*1e-3)

# ...existing code...
# ...existing code...

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors

# Calculate derived metrics
df["fps"] = 1.0 / (df["per_step_ms"]*1e-3)
df["rays_per_us"] = 1.0 / df["time_per_ray_us"]

# Define a consistent color map for resolution values
unique_res = sorted(df["resolution"].unique())
color_palette = plotly.colors.qualitative.Plotly
color_map = {res: color_palette[i % len(color_palette)] for i, res in enumerate(unique_res)}



# ...existing code...

# Define dash styles for modes
dash_map = {
    "single": "solid",
    "multi": "dash"
}
# ...existing code...

# Create subplots: 4 rows, 1 column
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        "Steps per ms vs Num Envs",
        "Rays per µs vs Num Envs",
        "Rays per µs vs Number of Rays",
        "Total Step Time (ms) vs Number of Rays"
    ],
    shared_xaxes=False
)

# ...existing code...

# 1. Steps per ms vs Num Envs
for res in unique_res:
    for mode in df["mode"].unique():
        sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["num_envs"],
                y=sub_df["fps"],
                mode="lines+markers",
                name=f"Res={res}, Mode={mode}",
                legendgroup=f"{res}-{mode}",
                showlegend=True,
                line=dict(color=color_map[res], dash=dash_map.get(mode, "solid"))
            ),
            row=1, col=1
        )

# 2. Rays per µs vs Num Envs
for res in unique_res:
    for mode in df["mode"].unique():
        sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["num_envs"],
                y=sub_df["rays_per_us"],
                mode="lines+markers",
                name=f"Res={res}, Mode={mode}",
                legendgroup=f"{res}-{mode}",
                showlegend=False,
                line=dict(color=color_map[res], dash=dash_map.get(mode, "solid"))
            ),
            row=2, col=1
        )

# 3. Rays per µs vs Number of Rays
for res in unique_res:
    for mode in df["mode"].unique():
        sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["total_rays"],
                y=sub_df["rays_per_us"],
                mode="lines+markers",
                name=f"Res={res}, Mode={mode}",
                legendgroup=f"{res}-{mode}",
                showlegend=False,
                line=dict(color=color_map[res], dash=dash_map.get(mode, "solid"))
            ),
            row=3, col=1
        )

# 4. Total Step Time (ms) vs Number of Rays
for res in unique_res:
    for mode in df["mode"].unique():
        sub_df = df[(df["resolution"] == res) & (df["mode"] == mode)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["total_rays"],
                y=sub_df["per_step_ms"],
                mode="lines+markers",
                name=f"Res={res}, Mode={mode}",
                legendgroup=f"{res}-{mode}",
                showlegend=False,
                line=dict(color=color_map[res], dash=dash_map.get(mode, "solid"))
            ),
            row=4, col=1
        )

fig.update_xaxes(title_text="Number of Environments", row=1, col=1)
fig.update_yaxes(title_text="Steps per ms", row=1, col=1)
fig.update_xaxes(title_text="Number of Environments", row=2, col=1)
fig.update_yaxes(title_text="Rays per µs", row=2, col=1)
fig.update_xaxes(title_text="Total Rays", row=3, col=1)
fig.update_yaxes(title_text="Rays per µs", row=3, col=1)
fig.update_xaxes(title_text="Total Rays", row=4, col=1)
fig.update_yaxes(title_text="Total Step Time (ms)", row=4, col=1)
fig.update_layout(
    height=1600,
    title_text="Raycast Benchmark Comparison",
    legend_title_text="Resolution & Mode"
)

fig.show()