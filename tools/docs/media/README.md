# Documentation media tools

Each documentation page with generated media has a `generate_<page>.sh` script.
Run a generator from anywhere in the repository; it writes the final assets
directly to `docs/source/_static`.

For the quickstart page:

```bash
tools/docs/media/generate_quickstart.sh
```

For the reinforcement-learning page:

```bash
tools/docs/media/generate_reinforcement_learning.sh
```

The reinforcement-learning generator trains its own Anymal-D policy progression. To reuse a
compatible completed run, set `RL_PROGRESS_CHECKPOINT_DIR` to a run directory containing
`model_0.pt`, `model_100.pt`, and `model_299.pt`.

The generators require `uv`, `ffmpeg`, a CUDA-capable GPU, and the optional
runtime packages installed by their `uv run --extra` commands.

## Renderer gallery

The renderer gallery compares Newton Warp, OVRTX, and Isaac RTX camera outputs.
Pass the shared editable USD stage when regenerating all RGB animations and
still output modes:

```bash
OMNI_KIT_ACCEPT_EULA=Y tools/docs/media/generate_renderer_gallery.sh \
    /path/to/renderer-gallery-scene.usda
```

The generator launches the kit-less renderers separately from Isaac RTX because
their optional runtime packages cannot share one process. It writes the final
WebP and PNG assets directly to `docs/source/_static/overview/sensors`.
