# Isaac Lab Policy Debug

`isaaclab_policy_debug` compares RSL-RL checkpoints from one run inside a single
Isaac Lab application. Each selected checkpoint controls one environment row,
while a Newton GL viewer shows the rows in a grid or at the same rendered
origin with translucent ghosts.

The package doesn't replace a task's action, observation, reward, or
termination managers. A scenario adapter owns synchronized reset behavior and
task-specific validation.

## Install

```bash
./isaaclab.sh -i policy_debug
```

## Run

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Pick-Place-Franka-Play-v0 \
  --policy-debug /absolute/path/to/rsl-rl-run
```

The run directory must contain checkpoints as direct `*.pt` children. New
files appear unchecked after their size and modification time remain unchanged
for two scans. Physics stays paused until at least one checkpoint is selected.

`--policy-debug-max-policies` sets the fixed environment capacity before Isaac
Lab builds the scene; its default is eight. Policy-debug mode rejects an
explicit checkpoint, `--num_envs`, and headless operation because those options
conflict with the interactive multi-policy session.

## Scenario adapters

The launcher resolves an adapter in this order: an explicit
`--policy-debug-adapter`, the Gym registration key
`policy_debug_adapter_entry_point`, then the built-in manager-based adapter.
The built-in adapter only runs a task when it can repeat and verify relative
scene state and command values across every active row.

Tasks with reset banks, continuing episodes, or state outside the standard
scene and command managers should implement `PolicyDebugScenarioAdapter` and
register the factory with the task. The adapter can also choose which scene
assets remain visible in translucent overlay layers and report a numerical
failure for one lane without closing the viewer.
