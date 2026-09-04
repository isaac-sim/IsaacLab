# {{ name }}

An installable downstream Isaac Lab task package generated with a standard uv `src` layout.

The registered tasks are:

{% for specification in specifications %}
- `{{ specification.task.id }}`
{% endfor %}

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
Newton physics with the OVRTX renderer. Commit `pyproject.toml` and `uv.lock` so collaborators use the same environment.

## Run the generated tasks

Replace the placeholders below with a task listed above and a selected RL library.

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

## Project structure

Task families live in `src/{{ name }}/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/{{ robot_name }}`. Add another robot by
creating a sibling of `config/{{ robot_name }}`; add another task by creating a sibling task-family directory.

## Development

Run the generated registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code, run the `setup_python_env` task or invoke its command directly:

```bash
uv run python .vscode/tools/setup_vscode.py
```

{% if include_ui_extension %}
## Isaac Sim UI extension

Add the project root to the Isaac Sim Extension Manager search paths, refresh, and enable the extension under
`Third Party`. Launch Isaac Sim through the project's `isaacsim` extra so the UI dependencies are available. Kit loads
`src/{{ name }}/ui_extension_example.py` through `config/extension.toml`.

{% endif %}
## Troubleshooting

If Pylance cannot resolve simulator modules, run the VS Code setup command above and reload the window. If indexing uses
too much memory, remove unused simulator extension paths from `.vscode/settings.json`.
