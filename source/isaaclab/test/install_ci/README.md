# Installation tests

Run these from a clean checkout without a local `_isaac_sim` build. Each runtime gets one
fresh uv environment, shared by its import and CLI checks. Wheel probes run outside the
checkout with Python environment variables cleared, so editable imports cannot mask missing
packaged resources. Commands have subprocess timeouts and report their captured output on failure.

The dependency-resolution check does not install packages or need a GPU:

```bash
uv run --no-project --with pytest python -m pytest source/isaaclab/test/install_ci/test_resolution.py
```

Run fresh workspace and wheel installs in the installation container:

```bash
uv run --no-project --with pytest python tools/run_install_ci.py docker --build-wheel
```

Add `--gpu` to enable short training checks. For native runs, pass `--run-gpu` after `--`.
Use `--wheel /path/to/isaaclab.whl` to reuse a built wheel, and pytest's `-k` or `-m resolve`
to select a contract. Fresh installation checks need network access and enough disk for both
Isaac Sim and CUDA packages; they are marked `slow`. Native commands use the same pytest suite.

Camera rendering and camera training remain in `misc/cartpole_training_smoke.py`, executed by
architecture CI in its prepared environment rather than reinstalling dependencies per probe.
