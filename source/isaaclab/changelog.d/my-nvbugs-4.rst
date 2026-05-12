Fixed
^^^^^

* Fixed ``ModuleNotFoundError: No module named 'isaaclab_physx'`` on kit-less /
  Newton-only installs (e.g. the ``newton,tasks,assets,ov,rl[rsl_rl]`` selector)
  by removing the remaining unconditional ``isaaclab_physx`` imports from
  ``isaaclab`` core:

  * :mod:`isaaclab.sim.spawners.meshes.meshes_cfg` and
    :mod:`isaaclab.sim.spawners.from_files.from_files_cfg` fall back to a dummy
    :class:`DeformableObjectSpawnerCfg` stub (with a warning) when
    ``isaaclab_physx`` is not installed, so ``MeshCfg`` / ``FileCfg`` and their
    subclasses remain importable.
  * :mod:`isaaclab.sim.spawners.meshes.meshes` and
    :mod:`isaaclab.sim.spawners.from_files.from_files` lazily import PhysX-only
    schemas, materials, and ``RigidBodyMaterialCfg`` inside the deformable-body
    and compliant-contact code paths, and use the solver-common
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` for
    rigid-material ``isinstance`` checks so rigid-only spawning works without
    ``isaaclab_physx``.
  * :attr:`~isaaclab.sim.SimulationCfg.physics_material` now uses a
    ``default_factory`` that prefers
    :class:`~isaaclab_physx.sim.spawners.materials.RigidBodyMaterialCfg` when
    ``isaaclab_physx`` is installed and falls back to
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` on
    kit-less / Newton-only installs, so importing :class:`SimulationCfg` no
    longer triggers the ``RigidBodyMaterialCfg`` forwarding shim.
