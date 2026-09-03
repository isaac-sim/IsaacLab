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
