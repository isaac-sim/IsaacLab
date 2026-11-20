* Added the ``simready`` object preset to the lift and reorient environments, which spawns 100 graspable
  SimReady props in place of the primitive shapes (``presets=simready``). The props come from a SimReady search
  in the environment configuration, whose result is cached in ``simready_objects.json``. ``Isaac-Lift-Franka``
  rejects the preset, since its pre-grasp reset covers only the primitive shapes.
