Added
^^^^^

* Added the ``ovphysx`` preset to ``Isaac-Repose-Cube-Allegro-Direct-v0``
  (``ObjectCfg`` and ``PhysicsCfg`` in
  :mod:`isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg`), so the
  task can be selected with ``presets=ovphysx`` against the OVPhysX
  backend.  Exercises the OVPhysX :class:`~isaaclab_ovphysx.assets.Articulation`
  (Allegro hand) and :class:`~isaaclab_ovphysx.assets.RigidObject` (cube)
  in the same scene.
