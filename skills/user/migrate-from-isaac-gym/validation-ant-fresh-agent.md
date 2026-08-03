# Initial Ant Smoke Validation

## Scope

A fresh agent used this skill to clone `https://github.com/isaac-sim/IsaacGymEnvs`, select the Ant locomotion task, and build a scratch Isaac Lab direct migration outside the Isaac Lab checkout. This is a historical smoke validation for a simple locomotion task; use [rough locomotion validation](validation-rough-locomotion.md) as the representative policy-success validation. For future validations, start scratch work from the Isaac Lab template generator's external project layout.

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

Runtime rerun with a compatible Isaac Sim runtime:

- The runtime used the PR checkout, Python 3.12.13, and a Kit 110-era Isaac Sim runtime.
- `isaacsim` and `omni` resolved from the active runtime. The `isaacsim.simulation_app` submodule was not present as a standalone import, but Isaac Lab's `AppLauncher` worked.
- A clean `PYTHONPATH` was required: the generated project extension first, then every package directory under this checkout's `source/`. Without this, Python mixed packages from another Isaac Lab checkout and hit duplicate Gym registrations.
- The migrated environment constructed with 4 environments on `cuda:0`, `reset()` returned observations with shape `(4, 60)`, and one random step returned observations with shape `(4, 60)`.
- A 2-iteration RSL-RL smoke run with 64 environments completed and wrote metrics. Iteration 0 logged mean reward `4.10`, mean episode length `13.00`, and success rate `1.0000`; iteration 1 logged mean reward `-19.77`, mean episode length `43.29`, and success rate `0.0000`.

Open gates:

- The scratch migration used a broad `ContactSensorCfg` pattern for foot observations and emitted missing rigid-body/contact-report warnings. Future migrations should validate sensor body names and use `JointWrenchSensorCfg` when legacy force-torque observations need torque components.
- Isaac Lab's random-agent entry point was not validated for this external migration package because it needs a wrapper or callback that imports the task before Gym lookup.
- The short training smoke proved that training starts and steps, but did not prove a successful Ant policy. Do not claim policy success until a longer run shows stable success or reward improvement.

## Skill Updates From This Validation

- Run runtime preflight through `uv run python` from the Isaac Lab checkout before promising reset, random-agent, or training success.
- Start external migrations from the Isaac Lab template generator, then put the generated project extension and all target checkout `source/` packages first on `PYTHONPATH` during external validation.
- Make external migration package registration explicit for scripts that do not expose `--external_callback`.
- Add legacy force/torque sensor mapping guidance so agents do not silently drop force sensor observations or ignore unresolved sensor prim warnings.

## Pass Criteria For A Complete Runtime Validation

For a complete validation, repeat the same migration and require:

1. The migrated task constructs with a small environment count.
2. `reset()` and several random `step()` calls succeed.
3. Isaac Lab's random-agent entry point runs against the migrated task.
4. A short training job starts and logs policy metrics.
5. Sensor prim paths and body names resolve without missing rigid-body or contact-report warnings.
6. A longer training job reaches the task's success criterion or shows clear reward improvement against the direct baseline.
