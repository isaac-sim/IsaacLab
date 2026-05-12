Fixed
^^^^^

* Fixed ``ModuleNotFoundError: No module named 'isaaclab_physx'`` on kit-less /
  Newton-only installs (e.g. the ``newton,tasks,assets,ov,rl[rsl_rl]`` selector)
  by making the ``isaaclab_physx`` imports in :mod:`isaaclab.sim.spawners.meshes`
  and :mod:`isaaclab.sim.spawners.from_files` optional. The ``*_cfg`` modules
  now fall back to a dummy :class:`DeformableObjectSpawnerCfg` stub (with a
  warning) when ``isaaclab_physx`` is not installed, and the spawn functions
  lazily import PhysX-only types inside the deformable-body code path so that
  rigid-only spawning works without ``isaaclab_physx``.
