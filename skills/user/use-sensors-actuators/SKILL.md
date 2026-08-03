---
name: isaaclab-using-sensors-actuators
description: Adds and configures Isaac Lab sensors, actuators, and sensor-derived task terms in new or existing tasks. Use when adding cameras, contact sensors, foot or feet contacts, contact history, touchdown timing, air-time rewards, undesired contacts, sensor observations or terminations, ray casters, IMUs, joint wrench sensors, tiled cameras, or actuator models.
audience: user
status: experimental
owners:
  - isaaclab-maintainers
---

# Using Sensors And Actuators

## When To Use

Use this skill when a user wants to add, configure, or debug sensors or actuator models in an Isaac Lab task.

Do not use this skill as a complete sensor or actuator catalog. Point to the API docs and maintained examples for available classes and parameters.

## Workflow

1. Identify the data needed by the task: contacts, ray hits, camera images, IMU data, joint wrench data, actuator dynamics, or low-dimensional robot state.
2. Read the relevant sensor or actuator docs before editing configs.
3. Check whether the sensor requires a simulation app, renderer, backend-specific support, or special update period.
4. Add sensor or actuator configs to the environment scene or asset config using existing task examples as templates.
5. Register the sensor in the scene setup for direct workflows, or in the scene config for manager-based workflows.
6. Add observation terms only after confirming the sensor data shape and device.
7. For an existing task, inspect its scene config and existing MDP terms before adding code. Make the smallest task-local config change and create a shared MDP module only when the term is genuinely reusable.
8. For contact-heavy tasks, verify body name patterns, history length, and whether `track_air_time=True` is required. Reuse maintained terms such as `mdp.feet_air_time` when they match the task.
9. For ray-cast terrain perception, verify mesh paths and terrain import setup.
10. For camera-based RL, start with small environment counts and confirm renderer memory behavior.
11. For multi-backend contact, ray, frame, IMU, PVA, or joint-wrench sensors, use the common config from `isaaclab.sensors`; the active physics backend selects the implementation automatically. Use backend-specific fields or configs only when the current API or maintained source example requires them.
12. For actuator changes, compare default joint names, limits, stiffness, damping, effort limits, and backend differences.

## Validation

Use this checklist:

1. Instantiate the environment with a small number of environments.
2. Confirm the sensor or actuator config binds to the intended prims or joints.
3. Step the simulation and inspect data shape, dtype, device, and update timing.
4. Confirm observations include the intended data and no stale values.
5. Run a short rollout or training smoke test after shape validation.

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with sensor and actuator docs under `docs/source/overview/core-concepts/`, sensor tutorials under `docs/source/tutorials/04_sensors/`, API docs under `docs/source/api/`, and maintained task examples under `source/isaaclab_tasks/isaaclab_tasks/`. If sensor support changes for PhysX, Newton, or renderers, update the docs or source examples first.

## References

- [Evaluations](evaluations.md)
- [Examples](examples.md)
- [Add sensors on robot tutorial](../../../docs/source/tutorials/04_sensors/add_sensors_on_robot.rst)
- [Contact sensor docs](../../../docs/source/overview/core-concepts/sensors/contact_sensor.rst)
- [Joint wrench sensor docs](../../../docs/source/overview/core-concepts/sensors/joint_wrench_sensor.rst)
- [Actuators docs](../../../docs/source/overview/core-concepts/actuators.rst)
- [Sensors API](../../../docs/source/api/lab/isaaclab.sensors.rst)
- [Sensor patterns API](../../../docs/source/api/lab/isaaclab.sensors.patterns.rst)
- [Actuators API](../../../docs/source/api/lab/isaaclab.actuators.rst)
