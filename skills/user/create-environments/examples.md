# Environment Creation Examples

## Manager-Based Workflow

Use manager-based workflow by default for new Isaac Lab tasks. This is the framework's main task-building path because observations, rewards, commands, events, curricula, and terminations can be reused and tuned independently.

Start from:

- `docs/source/tutorials/03_envs/create_manager_base_env.rst`
- `docs/source/tutorials/03_envs/create_manager_rl_env.rst`
- `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py`

For quadruped locomotion requests with custom command sampling or custom rewards, first try to model the behavior as reusable `CommandManager`, `RewardManager`, `ObservationManager`, and shared MDP functions. Inspect `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py` and robot-specific configs such as `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py`.

Smoke-test pattern:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole --num_envs 8
```

## Direct Workflow

Use direct workflow when the task has custom low-level step/reset logic, must stay close to a monolithic Isaac Gym task during migration, or is a performance prototype that will be converted later if it becomes reusable.

Start from:

- `docs/source/tutorials/03_envs/create_direct_rl_env.rst`
- `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env_cfg.py`

Smoke-test pattern:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Cartpole-Direct --num_envs 8
```

Training pattern:

```bash
uv run isaaclab train --rl_library rl_games --task Isaac-Cartpole-Direct
```

## Registration And Agent Configs

When adding a new Gym task:

1. Register the task with an `env_cfg_entry_point`.
2. Add one agent config entry point per supported RL framework.
3. Run `random_agent.py` before training.
4. Train only after action, observation, reward, reset, and termination shapes are stable.

Reference:

- `docs/source/tutorials/03_envs/register_rl_env_gym.rst`
- `docs/source/tutorials/03_envs/configuring_rl_training.rst`
