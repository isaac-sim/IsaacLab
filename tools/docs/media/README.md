# Documentation media tools

Each documentation page with generated media has its own folder. Run a page's
generator from anywhere in the repository; it writes the final assets directly
to `docs/source/_static`.

For the quickstart page:

```bash
tools/docs/media/quickstart/generate.sh
```

The generators require `uv`, `ffmpeg`, a CUDA-capable GPU, and the optional
runtime packages installed by their `uv run --extra` commands.
