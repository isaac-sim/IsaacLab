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
It uses the published Nucleus stage by default when regenerating all RGB
animations and still output modes:

```bash
OMNI_KIT_ACCEPT_EULA=Y tools/docs/media/generate_renderer_gallery.sh
```

Pass a local path or another Nucleus URI to override the default stage:

```bash
OMNI_KIT_ACCEPT_EULA=Y tools/docs/media/generate_renderer_gallery.sh \
    /path/to/renderer-gallery-scene.usda
```

The generator launches the kit-less renderers separately from Isaac RTX because
their optional runtime packages cannot share one process. It writes the final
WebP and PNG assets directly to `docs/source/_static/overview/sensors`.

The animated RGB clips and the albedo, Newton Warp shadow, and MDL
simple-shading stills are too large to keep in the repository. Upload them to
`https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/` after
regenerating; the renderer concept pages reference them by URL and do not track
them under `_static`.

## Checking published images

After uploading or syncing external images, run:

```bash
uv run python tools/docs/check_media.py docs/source/policy_deployment/05_leapp/exporting_policies_with_leapp.rst
```

The check performs GET requests and verifies image content types and GIF, PNG, JPEG, WebP or SVG
signatures without downloading the full assets. It makes up to three attempts, including 404
responses during upload/sync. Documentation CI warns for changed RST files on PRs and pushes;
the existing weekly and manual runs recheck all remote RST images and fail on persistent problems.
Use `--warn-only` for reminders locally, or omit paths to check all documentation images.
