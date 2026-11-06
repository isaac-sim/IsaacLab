# Sensors And Actuators Evaluations

## Scenario 1: Contact Sensor

Query: "Add foot contact observations and air-time rewards to my quadruped task."

Expected behavior:

- Loads `isaaclab-using-sensors-actuators` rather than the whole environment-building workflow.
- Points to contact sensor docs and nearby locomotion examples.
- Inspects the existing scene and MDP terms before adding code.
- Adds or updates a `ContactSensorCfg`, enabling `track_air_time` when needed, then wires the observation and existing `mdp.feet_air_time` reward term with the correct foot body pattern.
- Checks body name patterns, history length, update period, contact thresholds, data shape, and device.
- Validates sensor data shape before adding observations and rewards.

Known failure modes:

- Reads raw contact tensors directly without configuring the scene sensor.
- Uses body name patterns that do not match the robot asset.
- Creates a new shared-core observations module before checking the task's existing sensor and MDP terms.

## Scenario 2: Ray-Cast Terrain Perception

Query: "Add height measurements for rough terrain locomotion."

Expected behavior:

- Points to ray-caster usage in maintained rough locomotion examples.
- Checks terrain mesh paths and ray pattern configuration.
- Adds observation data only after validating ray hit shape and device.

Known failure modes:

- Adds height observations without binding the ray caster to the ground mesh.
- Assumes flat-terrain observation dimensions still apply.

## Scenario 3: Actuator Model

Query: "Replace ideal position control with a learned actuator model."

Expected behavior:

- Checks actuator docs, robot asset config, joint names, and backend constraints.
- Preserves joint limits and control semantics when changing actuator config.
- Runs a small rollout to catch stability or device issues.

Known failure modes:

- Changes actuator type without checking joint names and limits.
- Treats actuator behavior as independent of physics backend.

## Scenario 4: Camera Observation

Query: "Add image observations to Cartpole training."

Expected behavior:

- Points to camera-enabled Cartpole examples and sensor docs.
- Checks renderer requirements, data types, image size, and memory cost.
- Recommends a small environment count before training.
- Verifies the observation shape and framework support before changing agent configs.

Known failure modes:

- Adds camera tensors to observations without enabling a compatible renderer.
- Starts with thousands of environments before validating memory usage.
- Ignores whether the selected RL framework can consume the observation space.
