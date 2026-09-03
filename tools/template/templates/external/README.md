# {{ name }}

An external Isaac Lab project containing an installable Python package and Isaac Sim extension.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

Optional backends are available through extras. Pass the extra to each `uv run` command that needs it:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx
```

The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes. You can also select `ovrtx` independently when using
Newton physics with the OVRTX renderer.

Commit both `pyproject.toml` files and `uv.lock` so collaborators use the same environment.

## Run the generated tasks

Replace the placeholders below with a generated task and selected RL library.

```bash
# List this project's environments and their available presets
uv run python scripts/list_envs.py --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `list_envs.py`.

## Development

Run tests, formatting, and lint checks through the project environment. The
project registers the `unit`, `integration`, `smoke`, and `kitless` pytest
markers.

```bash
uv run pytest tests
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository
are not part of the installed `isaaclab_tasks` package. Keep test fixtures in
this project and use public Isaac Lab APIs. If you copy `env_test_utils.py`, it
becomes vendored code whose upstream changes you must track.

To configure VS Code, run the `setup_python_env` task or invoke its command directly:

```bash
uv run python .vscode/tools/setup_vscode.py
```

## Isaac Sim extension

Add the project's `source` directory to the Isaac Sim Extension Manager search paths, refresh, and enable the extension
under `Third Party`. The optional UI example is in `source/{{ name }}/{{ name }}/ui_extension_example.py`.
If you do not need the UI, delete that file and the matching `[[python.module]]` entry in
`source/{{ name }}/config/extension.toml`.

## Troubleshooting

If Pylance cannot resolve simulator modules, run the VS Code setup command above and reload the window. If indexing uses
too much memory, remove unused simulator extension paths from `.vscode/settings.json`.
