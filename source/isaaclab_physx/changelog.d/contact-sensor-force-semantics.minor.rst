Changed
^^^^^^^

* Renamed PhysX contact sensor normal and filtered-friction outputs to use the shared explicit
  force names. Aggregate friction remains unsupported; use ``friction_force_matrix_w`` with
  configured filter objects. ``net_forces_w`` is the total contact force; PhysX cannot compute
  it, so the property returns ``net_normal_forces_w`` and warns. ``friction_forces_w`` is the
  aggregate friction force; PhysX only provides filtered friction, so the property returns
  ``friction_force_matrix_w`` and warns (known limitations planned to be fixed in a later
  release).
* Added ``friction_force_matrix_w_history`` for filtered friction force history.
