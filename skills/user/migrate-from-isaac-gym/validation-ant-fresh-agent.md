# Fresh Ant Migration Validation

## Scope

A fresh agent used this skill to clone `https://github.com/isaac-sim/IsaacGymEnvs`, select the Ant locomotion task, and build a scratch Isaac Lab direct migration outside the Isaac Lab checkout.

Source artifacts:

- `isaacgymenvs/tasks/ant.py`
- `isaacgymenvs/cfg/task/Ant.yaml`
- `assets/mjcf/nv_ant.xml`

The scratch migration registered a direct task named `IsaacGym-Ant-Migrated-Direct-v0` with an Isaac Lab Ant analogue, PhysX settings, a 60-dimensional policy observation, contact sensor inputs, and RSL-RL / RL-Games training configs.

## Results

Passed gates:

- Cloned IsaacGymEnvs.
- Imported and registered the migrated task.
- Resolved environment and agent configs.
- Statically compiled the migrated package.

Blocked gates:

- Environment construction, reset, random-agent execution, and training did not start on the validation machine because the local simulator runtime was incompatible with the Isaac Lab 3.0 checkout.
- The active `./isaaclab.sh -p` Python did not provide `isaacsim`, `isaacsim.simulation_app`, or `omni`.
- The available Isaac Sim runtime was 4.2 / Python 3.10 / Kit 106-era, while this checkout required a compatible Python 3.12 / Kit 110-era runtime.
- The RSL-RL attempt also found an older local `rsl_rl` package missing `DistillationRunner`.

## Skill Updates From This Validation

- Run runtime preflight before promising reset, random-agent, or training success.
- Make external scratch package registration explicit for scripts that do not expose `--external_callback`.
- Add legacy force/torque sensor mapping guidance so agents do not silently drop force sensor observations.

## Pass Criteria For A Complete Runtime Validation

On a compatible Isaac Sim runtime, repeat the same migration and require:

1. The migrated task constructs with a small environment count.
2. `reset()` and several random `step()` calls succeed.
3. Isaac Lab's random-agent entry point runs against the migrated task.
4. A short training job starts and logs policy metrics.
5. A longer training job reaches the task's success criterion or shows clear reward improvement against the direct baseline.
