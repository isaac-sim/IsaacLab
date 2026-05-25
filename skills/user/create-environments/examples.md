# Environment Creation Examples

## Direct Workflow

Use direct workflow when the task has custom step logic or is being migrated from a monolithic task.

Start from:

- `docs/source/tutorials/03_envs/create_direct_rl_env.rst`
- `source/isaaclab_tasks/isaaclab_tasks/direct/`

Smoke-test pattern:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 8
```

Training pattern:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Cartpole-Direct-v0 --headless
```

## Manager-Based Workflow

Use manager-based workflow when observations, rewards, commands, events, or terminations should be reusable.

Start from:

- `docs/source/tutorials/03_envs/create_manager_base_env.rst`
- `docs/source/tutorials/03_envs/create_manager_rl_env.rst`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/`

Smoke-test pattern:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 8
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
