# Installing kitless XR

Use Python 3.12 on Linux x86-64. The Isaac Capture wheel includes the
OVXR session integration, native camera-panel helper and desktop mirror modules.

Complete the [CloudXR, client, network and license prerequisites](kitless_xr.md#prerequisites)
before starting a session.

From the Isaac Lab root, select the Capture version in an `ovxr` project
extra. Lab already configures NVIDIA's public Python index and allows prereleases:

```bash
uv add --optional ovxr --frozen \
    "isaaccapture[ovxr,retargeters,ui,cloudxr,desktop-mirror]==1.7.93a1 ; platform_system == 'Linux' and platform_machine == 'x86_64'"
uv add --optional ovxr --frozen \
    "isaaclab-dev[ov] ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "isaaclab-teleop ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "isaaclab-mimic ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "ipywidgets>=8.1.5 ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "dex-retargeting==0.5.0 ; platform_system == 'Linux' and platform_machine == 'x86_64'"
```

The `ov` extra supplies the compatible OVRTX, OVStage, OVPhysX and PhysX USD
schemas. Keep its provider requirements and the workspace package metadata unchanged.

Add these mutually exclusive selections to `tool.uv.conflicts`, preserving existing entries:

```toml
[tool.uv]
conflicts = [
    [{ extra = "ovxr" }, { extra = "teleop" }],
    [{ extra = "ovxr" }, { extra = "isaacsim" }],
    [{ extra = "ovxr" }, { extra = "viser" }],
    [{ extra = "ovxr" }, { extra = "all" }],
]
```

The `teleop` extra selects Capture packages for Kit and non-Kit input
workflows. Use the separate `ovxr` selection for the XR commands here.
Isaac Sim and Viser require older WebSocket versions than this Capture
release, so these selections cannot be combined. The `all` extra
includes Viser. The wheel requirement includes the optional `desktop-mirror` extra;
omit that extra only when running with `--headless`. OVXR live and MCAP replay
open the mirror by default. The mirror requires Python 3.12 or newer and a
usable GPU-backed desktop display.

Resolve the selected environment and check its dependencies:

```bash
uv lock --refresh-package isaaccapture --upgrade-package isaaccapture \
    --upgrade-package numpy==2.3.5 --upgrade-package packaging==23.2
uv sync --python 3.12 --locked --extra ovxr
uv run --extra ovxr python -c 'import importlib.metadata as m; print(m.version("isaaccapture"), m.version("ovxr"))'
uv pip check
```

The lock selects compatible NumPy 2.3.5 and packaging 23.2 versions without
changing project-wide overrides. OVPhysX declares `packaging<24`; the explicit
selection above satisfies its metadata in this non-Kit selection. Stop applications before changing selections.
The Kit and OVXR provider selections are different; preserve the source metadata
and lockfile when changing between them, and check the resulting dependency graph.
Repeat this selection if a later lock upgrade replaces it. Isaac Capture's
`ovxr` extra selects OVXR and its compatible OVRTX/OVStage dependencies; see its
OVXR guide for application version overrides.

Continue with [live teleoperation](kitless_xr.md#live-teleoperation).
