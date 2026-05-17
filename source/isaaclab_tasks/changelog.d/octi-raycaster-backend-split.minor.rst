Added
^^^^^

* Added raycaster-camera depth presets (``raycaster_depth64``, ``raycaster_depth128``,
  ``raycaster_depth256``) for both base and wrist views in the Dexsuite Kuka-Allegro
  manipulation task, backed by
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCamera`. Targets the table,
  ground plane, manipulated object, and robot visuals.
