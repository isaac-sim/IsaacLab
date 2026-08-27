# Documentation media tools

Each documentation page with generated media has a `generate_<page>.sh` script.
Run a generator from anywhere in the repository; it writes the final assets
directly to `docs/source/_static`.

For the quickstart page:

```bash
tools/docs/media/generate_quickstart.sh
```

The generators require `uv`, `ffmpeg`, a CUDA-capable GPU, and the optional
runtime packages installed by their `uv run --extra` commands.

## Renderer gallery

The renderer gallery compares Newton Warp, OVRTX, and Isaac RTX camera outputs
using the editable `renderer_gallery_scene.usda` stage. Regenerate all RGB
animations and still output modes from the repository root with:

```bash
OMNI_KIT_ACCEPT_EULA=Y tools/docs/media/generate_renderer_gallery.sh
```

The generator launches the kit-less renderers separately from Isaac RTX because
their optional runtime packages cannot share one process. It writes the final
WebP and PNG assets directly to `docs/source/_static/overview/sensors`.
