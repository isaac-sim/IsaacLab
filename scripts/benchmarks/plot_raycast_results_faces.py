# %% Imports
import io
import pandas as pd
import numpy as np
import plotly.express as px




#     mode  num_envs  resolution  rays_per_sensor  total_rays  per_step_ms   avg_memory  num_faces  track_mesh_transforms  num_assets
# 0  multi       512        0.05            10201     5222912   138.751815  5672.463750        320                  False           1
# 1  multi       512        0.05            10201     5222912   138.627011  5592.880313       1280                  False           1
# 2  multi       512        0.05            10201     5222912   139.396509  5540.730938       5120                  False           1
# 3  multi       512        0.05            10201     5222912   139.861900  5537.465000      20480                  False           1
# 4  multi       512        0.05            10201     5222912   140.231446  5532.323125      81920                  False           1
# 5  multi       512        0.05            10201     5222912   140.318203  5563.559688     327680                  False           1


# %% Raw CSV (paste your data here) – OR replace with: df = pd.read_csv("your_file.csv")
df = pd.read_csv("/workspace/isaaclab/outputs/benchmarks/ray_caster_benchmark_n_faces.csv")

print(df.head())
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

# ...existing code...

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors

# Calculate derived metrics
df["fps"] = 1.0 / (df["per_step_ms"]*1e-3)
df["rays_per_us"] = 1.0 / df["time_per_ray_us"]

# Create subplots: 2 rows, 1 column
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "Average Memory vs Number of Faces",
        "Per Step Time vs Number of Faces"
    ],
    shared_xaxes=False
)

# 1. Average Memory vs Number of Faces
fig.add_trace(
    go.Scatter(
        x=df["num_faces"],
        y=df["avg_memory"],
        mode="markers+lines",
        name="Avg Memory"
    ),
    row=1, col=1
)

# 2. Per Step Time vs Number of Faces
fig.add_trace(
    go.Scatter(
        x=df["num_faces"],
        y=df["per_step_ms"],
        mode="markers+lines",
        name="Per Step Time"
    ),
    row=2, col=1
)

fig.update_xaxes(title_text="Number of Faces", row=1, col=1)
fig.update_yaxes(title_text="Average Memory (MB)", row=1, col=1)
fig.update_xaxes(title_text="Number of Faces", row=2, col=1)
fig.update_yaxes(title_text="Per Step Time (ms)", row=2, col=1)

fig.update_layout(
    height=800,
    title_text="Raycast Benchmark: Memory and Step Time vs Number of Faces"
)

fig.show()