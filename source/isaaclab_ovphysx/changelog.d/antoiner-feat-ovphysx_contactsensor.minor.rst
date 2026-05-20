Added
^^^^^

* Added :class:`~isaaclab_ovphysx.sensors.ContactSensor`,
  :class:`~isaaclab_ovphysx.sensors.ContactSensorCfg`, and
  :class:`~isaaclab_ovphysx.sensors.ContactSensorData` for the OVPhysX
  backend, satisfying the
  :class:`~isaaclab.sensors.contact_sensor.BaseContactSensor` and
  :class:`~isaaclab.sensors.contact_sensor.BaseContactSensorData`
  contracts. Wires net contact forces and the per-partner force matrix
  through the OVPhysX :class:`ovphysx.api.ContactBinding` API
  (``read_net_forces`` / ``read_force_matrix``); optional pose tracking
  reads through a ``RIGID_BODY_POSE`` :class:`ovphysx.api.TensorBinding`.
  Air/contact time tracking,
  :meth:`~isaaclab_ovphysx.sensors.ContactSensor.compute_first_contact`,
  :meth:`~isaaclab_ovphysx.sensors.ContactSensor.compute_first_air`,
  history buffers, and reset semantics mirror the PhysX backend.
* Added the shared
  :mod:`isaaclab_ovphysx.sensors.kernels` module with
  :func:`~isaaclab_ovphysx.sensors.kernels.concat_pos_and_quat_to_pose_kernel`
  and the 1D variant for reuse across future OVPhysX sensors.

Changed
^^^^^^^

* Changed the existing
  ``source/isaaclab_ovphysx/test/sensors/check_contact_sensor.py``
  stubs to real tests adapted from the PhysX
  :mod:`isaaclab_physx.test.sensors.test_contact_sensor` suite. The
  three tests that exercise ``track_contact_points`` or
  ``track_friction_forces`` are decorated with
  :func:`pytest.mark.skip` until the OVPhysX wheel ships
  tensor-friendly per-sensor reads (see
  ``docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md``);
  the test bodies are preserved so the decorator can be removed in a
  follow-up.

Removed
^^^^^^^

* **Breaking:** Removed the five
  ``source/isaaclab_ovphysx/test/sensors/check_contact_sensor.py``
  ``pytest.skip("Contact sensor not yet supported by ovphysx
  backend.")`` placeholders in favour of the real test suite above.
  No public migration is required; the placeholder names did not
  appear in any external API.
