# Installing kitless XR

Use Python 3.12 on Linux x86-64. The checked-in `ovxr` extra and lockfile select
Isaac Capture 1.7.118rc1 with the OVXR session owner, camera panels and desktop mirror,
and the compatible OVRTX, OVStage and OVPhysX packages. No local metadata edits are needed.

Complete the [CloudXR, client, network and license prerequisites](kitless_xr.md#prerequisites)
before starting a session. From the Isaac Lab root:

```bash
uv sync --python 3.12 --locked --extra ovxr
uv run --no-sync python -c 'from isaaccapture.ovxr_session_manager import OvxrSession; import importlib.metadata as m; print(m.version("isaaccapture"), m.version("ovxr"))'
```

The `ovxr` extra includes `teleop` and uses the same Capture release as Kit input
workflows. It can be installed alongside `isaacsim` and `viser`. Stop applications
before changing selections. Keep the repository's dependency overrides: Capture's
CloudXR extra requires WebSockets 17.3 or newer, overriding Isaac Sim 6.2's `<15`
and Viser 1.0's `<17` bounds. The packaging override also reconciles OVPhysX's older
cap with the rest of the stack.
Do not downgrade packaging independently or modify the lockfile during installation.

The desktop mirror is optional at runtime. Enable it with `--ovxr_desktop_mirror`
on a host with a usable GPU-backed desktop display. Without that option, no local
window is opened; headset rendering still requires the GPU. Camera panels are
independent of the mirror.

Add `--enable_debug_visualization` to show red hand-joint markers and controller aim axes
when their tracking data is available. These markers appear in the headset eye views and their
desktop-mirror views, but are excluded from sensor images, including PiP. They do not require
an ordinary desktop visualizer.

For adapter regression tests, include the test tools:

```bash
uv sync --python 3.12 --locked --extra ovxr --extra test
uv run --no-sync python -m pytest source/isaaclab_teleop/test/test_ov_libraries_xr.py
uv run --no-sync python -m pytest source/isaaclab_teleop/test/test_ovxr_session.py
```

The ordinary Kit dependency selection cannot qualify these tests.
Native GPU, desktop and headset qualification are separate from these contract tests.

Continue with [live teleoperation](kitless_xr.md#live-teleoperation).
