Changed
^^^^^^^

* Moved the physics-engine event-randomization terms (:func:`~isaaclab.envs.mdp.randomize_rigid_body_material`,
  :func:`~isaaclab.envs.mdp.randomize_rigid_body_mass`, :func:`~isaaclab.envs.mdp.randomize_rigid_body_inertia`,
  :func:`~isaaclab.envs.mdp.randomize_rigid_body_com`, :func:`~isaaclab.envs.mdp.randomize_rigid_body_collider_offsets`,
  :func:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`, :func:`~isaaclab.envs.mdp.randomize_joint_parameters`,
  :func:`~isaaclab.envs.mdp.randomize_fixed_tendon_parameters`, :func:`~isaaclab.envs.mdp.randomize_actuator_gains`,
  and :func:`~isaaclab.envs.mdp.randomize_rigid_body_scale`) into the new
  ``isaaclab.envs.mdp.physics_events`` module. Existing imports from
  ``isaaclab.envs.mdp.events`` and :mod:`isaaclab.envs.mdp` remained valid.
  Backend-specific implementations are resolved at runtime from the active physics manager.
