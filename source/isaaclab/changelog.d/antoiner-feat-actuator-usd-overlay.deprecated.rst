* **Breaking:** Deprecated ``isaaclab.sim.schemas.schemas_actuators.resolve_per_dof`` and retained a
  functional wrapper using strict joint-name pattern resolution. Unmatched and overlapping patterns
  now raise ValueError. For regex mappings, migrate to
  :func:`~isaaclab.utils.string.resolve_matching_names_values`; handle None as an empty mapping and
  broadcast scalars directly.
