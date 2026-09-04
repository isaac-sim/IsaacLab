# {{ name }}

An external Isaac Lab project containing an installable Python package and Isaac Sim extension.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment:

```bash
uv sync
```

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

Run formatting and lint checks through the project environment:

```bash
uv run pre-commit run --all-files
```

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run python .vscode/tools/setup_vscode.py
```

The setup command selects the active interpreter and generates a git-ignored `pyrightconfig.json`. The generated
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project package,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

For an Isaac Sim binaries installation that is not available in the project environment, run the setup with its Python
launcher instead:

```bash
# Linux
<isaac-sim-path>/python.sh .vscode/tools/setup_vscode.py --isaac_path <isaac-sim-path>

# Windows
<isaac-sim-path>\python.bat .vscode\tools\setup_vscode.py --isaac_path <isaac-sim-path>
```

## Isaac Sim extension

Add the project's `source` directory to the Isaac Sim Extension Manager search paths, refresh, and enable the extension
under `Third Party`. The optional UI example is in `source/{{ name }}/{{ name }}/ui_extension_example.py`.

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
