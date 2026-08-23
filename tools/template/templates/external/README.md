# {{ name }}

An external Isaac Lab project generated for Isaac Lab 3.0. The project is an installable Python package and an
Isaac Sim extension, while remaining independent of the Isaac Lab source repository.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment:

```bash
uv sync
```

The generated package depends on Isaac Lab and the RL libraries selected during generation. The first sync can take
several minutes because uv downloads the simulator and learning dependencies. Commit both `pyproject.toml` files and
`uv.lock` so collaborators resolve the same environment.

## Run the generated tasks

The package advertises its tasks to the installed Isaac Lab CLI, so no project-specific runner scripts are needed.
Replace the placeholders below with one of the generated task and library combinations.

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

Use `physics=<PRESET>` to select a generated physics preset. Run the environment listing command above before choosing
a preset; the generated tasks expose Isaac Sim PhysX, automatic PhysX, OVPhysX, Newton MJWarp, and Newton Kamino
variants.

## Development

Run formatting and lint checks through the project environment:

```bash
uv run pre-commit run --all-files
```

To configure VS Code, run the `setup_python_env` task or invoke its command directly:

```bash
uv run python .vscode/tools/setup_vscode.py
```

## Isaac Sim extension

The Python package can also be loaded as an Isaac Sim extension. Add this project's `source` directory to the Extension
Manager search paths, refresh the manager, find the extension under `Third Party`, and enable it. The optional UI example
is implemented in `source/{{ name }}/{{ name }}/ui_extension_example.py`.

## Troubleshooting

If Pylance cannot resolve simulator modules, run the VS Code setup command above and reload the window. If indexing uses
too much memory, remove unused simulator extension paths from `.vscode/settings.json`.
