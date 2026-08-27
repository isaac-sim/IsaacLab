Changed
^^^^^^^

* Renamed PhysX contact sensor normal and filtered-friction outputs to use the shared explicit
  force names. Aggregate friction remains unsupported; use ``friction_force_matrix_w`` with
  configured filter objects.
