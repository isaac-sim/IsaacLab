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

## Environment preview images

Capture the environment browser's JPEG thumbnails with Kit using the checked-in
task and camera manifest:

```bash
OMNI_KIT_ACCEPT_EULA=Y uv run --extra isaacsim --extra tetrahedralization \
    python tools/docs/media/capture_environment_previews.py \
    --manifest tools/docs/media/environment_previews.json
```

Each task runs in its own process, resets with a fixed seed, and warms up rendering
without taking physics steps. Captures preserve each image's aspect ratio and
limit the longest edge to 960 pixels. The manifest also retains task-specific
camera framing and physics selections. Logs and receipts under
`logs/environment-preview-refresh` record the image and ground-texture hashes.

Use `--only anymal_d_flat.jpg` to recapture one image. To distribute the work
across machines or GPUs, pass `--shard 0/2` and `--shard 1/2` with separate
`--log-dir` directories. `--skip-existing` resumes successful captures after
checking their settings and image hashes. Use a new log directory for a later
asset refresh.

When checking a newly published asset before its CDN cache expires,
`--direct-s3-origin` reads the public S3 origin for that process. It does not
change persisted SDK settings.

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
