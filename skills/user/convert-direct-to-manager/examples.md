# Direct To Manager Examples

## Ant Locomotion Pair

Use the maintained Ant pair as the first comparison point:

- Direct baseline: `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_direct_env.py`
- Direct config: `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_direct_env_cfg.py`
- Manager config: `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py`

Important mappings:

| Direct source | Manager-based target |
| --- | --- |
| `_setup_scene()` and direct terrain config | `AntSceneCfg` with terrain, robot, sensors, and lights |
| `_apply_action()` joint efforts | `ActionsCfg` with `JointEffortActionCfg` |
| `_get_observations()` tensor concatenation | `ObservationsCfg.PolicyCfg` with ordered `ObsTerm` entries |
| `_get_rewards()` and reward helpers | `RewardsCfg` with `RewTerm` entries and shared MDP functions/classes |
| `_get_dones()` | `TerminationsCfg` with `DoneTerm` entries |
| `_reset_idx()` state reset | `EventCfg` reset terms |
| Physics variants in direct cfg | `PresetCfg` physics variants in manager cfg |

## Conversion Checklist

- Preserve the direct environment until the manager-based task passes smoke tests.
- Keep the policy observation contract stable unless the user explicitly wants to change policy inputs.
- Put reusable math and stateful reward logic in a task `mdp/` package.
- Use `SceneEntityCfg` to bind MDP terms to assets, joints, bodies, and sensors.
- Prefer parameter changes in config classes over custom environment subclasses.

## Smoke-Test Commands

Use suffixless task names for canonical manager-based tasks:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Ant --num_envs 8
uv run isaaclab train --rl_library rsl_rl --task Isaac-Ant --num_envs 64 --max_iterations 5
```

Keep the direct baseline available for comparison:

```bash
uv run python scripts/environments/random_agent.py --task Isaac-Ant-Direct --num_envs 8
```
