Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_newton.actuators.NewtonActuatorAdapter` to a flat
  DOF layout so heterogeneous scenes (different articulations per environment) are supported.
  The constructor now takes keyword-only ``dof_count`` (total flat DOF count of the model) and
  ``dof_env_id`` (per-DOF environment index) instead of ``num_joints`` and ``dof_offset``;
  :meth:`~isaaclab_newton.actuators.NewtonActuatorAdapter.bind_articulation` now takes a
  ``dof_index_map`` tensor of shape ``(num_instances, num_joints)`` instead of ``dof_offset``
  and ``num_joints``; and :func:`~isaaclab_newton.actuators.build_newton_actuator_defaults`
  now gathers from the adapter's flat gain and managed-DOF snapshots through ``dof_index_map``
  instead of filtering actuators by an environment-zero DOF window. For a homogeneous scene,
  migrate by passing ``dof_count=num_envs * num_joints``, a ``dof_env_id`` table mapping each
  flat DOF to its environment, and
  ``dof_index_map=torch.arange(num_instances * num_joints).reshape(num_instances, num_joints)``.

Added
^^^^^

* Added a build-time one-writer-per-DOF check to
  :class:`~isaaclab_newton.actuators.NewtonActuatorAdapter`: overlapping actuator index sets
  now raise a ``ValueError`` at construction instead of silently corrupting efforts through
  order-dependent scatters.

Fixed
^^^^^

* Fixed redundant whole-model actuator-state resets in
  :meth:`~isaaclab_newton.actuators.NewtonActuatorAdapter.reset`: a scene reset chain calls
  the model-global reset once per articulation with the same environment set, so identical
  repeats within a single reset event are now skipped.
