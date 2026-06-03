Changed
^^^^^^^

* Moved the physics-engine event-randomization terms (:func:`~isaaclab.envs.mdp.randomize_rigid_body_material`,
  :func:`~isaaclab.envs.mdp.randomize_rigid_body_mass`, :func:`~isaaclab.envs.mdp.randomize_rigid_body_inertia`,
  :func:`~isaaclab.envs.mdp.randomize_rigid_body_com`, :func:`~isaaclab.envs.mdp.randomize_rigid_body_collider_offsets`,
  :func:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`, :func:`~isaaclab.envs.mdp.randomize_joint_parameters`,
  :func:`~isaaclab.envs.mdp.randomize_fixed_tendon_parameters`, :func:`~isaaclab.envs.mdp.randomize_actuator_gains`,
  and :func:`~isaaclab.envs.mdp.randomize_rigid_body_scale`) out of ``isaaclab.envs.mdp.events`` into the new
  ``isaaclab.envs.mdp.physics_events`` module. They remain importable from :mod:`isaaclab.envs.mdp`; update any
  direct ``from isaaclab.envs.mdp.events import ...`` of these terms to import from :mod:`isaaclab.envs.mdp` (or
  ``isaaclab.envs.mdp.physics_events``). The backend-specific implementations are resolved at runtime from
  ``isaaclab_<backend>.envs.mdp`` based on the active physics manager.

Deprecated
^^^^^^^^^^

* Deprecated :func:`~isaaclab.envs.mdp.randomize_rigid_body_scale`. It is only supported on the PhysX backend
  (Newton raises :class:`NotImplementedError`); prefer multi-asset spawning with per-scale USD variants via
  :class:`~isaaclab.sim.MultiAssetSpawnerCfg`.
