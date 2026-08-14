# Newton Contact Force Matrix History Design

## Goal

Provide `ContactSensorData.force_matrix_w_history` on the Newton backend with
the same public shape, ordering, and availability semantics as PhysX. This lets
consumers compute per-(sensor, filtered-object) contact-force peaks across a
decimation window instead of observing only the last physics substep.

## Behavior

When a contact sensor has filtered counterparts, Newton will expose
`force_matrix_w_history` as a Warp `wp.vec3f` array with shape
`(num_envs, effective_history_length, num_sensors, num_filter_objects)`. Its
Torch view therefore has shape
`(num_envs, effective_history_length, num_sensors, num_filter_objects, 3)`.

The effective history length is `max(history_length, 1)`, matching PhysX. The
first history index contains the newest sample and the last index contains the
oldest. When no filtered counterparts exist, both `force_matrix_w` and
`force_matrix_w_history` remain `None`.

Newton's net-force history will use the same effective history length so the two
history properties retain matching backend semantics when `history_length` is
zero.

## Data Flow

`ContactSensorData.create_buffers()` allocates the current and historical force
matrices together whenever Newton reports one or more filtered counterparts.
Each sensor update copies Newton's current per-pair force matrix into Isaac Lab's
owned buffer, shifts older history samples by one position, and writes the new
sample at history index zero. Environment masks apply to both the current force
and its history, so an environment that is not updated does not advance.

Sensor reset clears current net forces, net-force history, the current force
matrix, and force-matrix history for only the selected environments. This keeps
contact peaks from a previous episode out of the next episode's reward window.

## Implementation Scope

The change stays within the Newton contact sensor:

- allocate and expose the history buffer in `contact_sensor_data.py`;
- extend the Newton contact-sensor kernels to roll and reset force-matrix
  history;
- pass the effective history length and history buffer through
  `contact_sensor.py`;
- add focused Newton tests and one `isaaclab_newton` changelog fragment.

No public API names or configuration fields change, and no dependency is added.

## Testing

Focused tests will verify:

- filtered sensors allocate both current and historical force matrices with the
  PhysX-compatible shape, including a one-sample history when configured length
  is zero;
- repeated distinct samples produce newest-first rolling history;
- masked environments retain their previous history;
- reset clears current and historical values only for selected environments;
- sensors without filtered counterparts continue to return `None`.

The regression test will be run before the production change to demonstrate the
current missing-history failure, then after the change to demonstrate the fix.
The relevant Newton sensor tests and repository pre-commit hooks will provide
final verification.
