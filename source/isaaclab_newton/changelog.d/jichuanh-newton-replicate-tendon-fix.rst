Fixed
^^^^^

* Fixed per-environment string identifiers (e.g. ``mujoco:tendon_label``)
  keeping the source proto path after replication.
  :func:`~isaaclab_newton.cloner.newton_replicate._rename_builder_labels`
  now also walks string-typed custom-attribute columns whose frequency
  declares a ``references="world"`` companion, rewriting their per-row
  source-path prefix to the destination world root in the same pass that
  handles built-in label arrays. Adds ``constraint_mimic`` and
  ``equality_constraint`` to that built-in pass for completeness. The
  prefix match uses a path-separator boundary so a source path that is a
  string prefix of another (e.g. ``/Sources/protoA`` vs
  ``/Sources/protoAB``) does not cross-contaminate during the rename.
