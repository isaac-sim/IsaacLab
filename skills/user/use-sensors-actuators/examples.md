# Sensors And Actuators Examples

## Contact Sensors

Use contact sensors for feet air time, undesired contacts, grasp checks, and termination conditions.

Examples to inspect:

- `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`

For multi-backend tasks, follow the `VelocityEnvContactSensorCfg` pattern: wrap PhysX, Newton, and OvPhysX contact sensor configs in a `PresetCfg` instead of assuming the base contact sensor cfg works identically on every backend.

Validation checklist:

- Body name patterns match the asset.
- `history_length` is sufficient for the reward or termination term.
- Sensor update period matches the control and simulation step.
- Contact data shape is checked before adding observations or rewards.

## Ray-Cast Height Scanners

Use ray casters for terrain perception on rough locomotion tasks.

Examples to inspect:

- `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`

Validation checklist:

- The scanner is mounted on the intended robot body.
- `mesh_prim_paths` points to the terrain mesh.
- Observation dimensions are updated for the ray pattern.
- Flat-terrain configs remove or disable height scan observations.

## Cameras

Use camera or tiled-camera examples before adding visual observations to training.

Examples to inspect:

- `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_camera_env_cfg.py`
- `docs/source/tutorials/04_sensors/add_sensors_on_robot.rst`

Validation checklist:

- Renderer requirements are met.
- Data types and image sizes fit memory limits.
- Training starts with a small number of environments.

## Actuators

Use actuator configs in robot assets or task overrides, then validate joint names and limits.

Examples to inspect:

- `source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/shadow_hand/shadow_hand_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/cabinet/cabinet_env_cfg.py`

Validation checklist:

- `joint_names_expr` matches the intended joints.
- Effort, velocity, stiffness, damping, and armature are compatible with the robot.
- Backend-specific behavior is checked before adding presets.
