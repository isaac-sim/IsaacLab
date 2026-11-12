# Installing kitless XR

Use Python 3.12 on Linux x86-64. The checked-in `ovxr` extra and lockfile select
Isaac Capture 1.7.93a1 with the OVXR session owner, camera panels and desktop mirror,
and the compatible OVRTX, OVStage and OVPhysX packages. No local metadata edits are needed.

Complete the CloudXR workstation, client, network and license prerequisites before
starting a session. From the Isaac Lab root:

```bash
uv sync --python 3.12 --locked --extra ovxr
uv run --no-sync python -c 'from isaaccapture.ovxr_session_manager import OvxrSession; import importlib.metadata as m; print(m.version("isaaccapture"), m.version("ovxr"))'
```

The `teleop` extra preserves the Capture selection for Kit input workflows. The
`ovxr` selection is mutually exclusive with `teleop`, `isaacsim`, `viser`, and
`all`, whose Capture or WebSocket requirements differ. Stop applications before
changing selections. Keep the repository's dependency overrides: in particular,
current main reconciles OVPhysX's older packaging cap with the rest of the stack.
Do not downgrade packaging independently or modify the lockfile during installation.

For adapter regression tests, include the test tools:

```bash
uv sync --python 3.12 --locked --extra ovxr --extra test
uv run --no-sync python -m pytest source/isaaclab_teleop/test/test_ov_libraries_xr.py
uv run --no-sync python -m pytest source/isaaclab_teleop/test/test_ovxr_session.py
```

The dedicated OVXR CI job checks the Capture owner import and rejects skipped
adapter tests. The ordinary Kit dependency selection cannot qualify these tests.
Native GPU, desktop and headset qualification are separate from these contract tests.
