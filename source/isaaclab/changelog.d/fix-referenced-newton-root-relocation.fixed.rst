* Fixed ``fix_root_link=True`` raising ``RuntimeError: Failed to remove '...'`` in
  :func:`~isaaclab.sim.schemas.modify_articulation_root_properties` when the articulation root link comes from a
  referenced USD layer (e.g. ``MjcfConverter``/``UrdfConverter`` output) with authored ``NewtonArticulationRootAPI``
  properties. The stage's edit target cannot remove reference-layer opinions, so the writer now logs a warning and
  completes the relocation: the properties stay flattened onto the parent and the schema is stripped from the former
  root, matching the behavior of the shared ``_relocate_articulation_root`` helper.
