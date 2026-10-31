* Changed the gym entry point of ``IsaacContrib-PickPlace-GR1T2-Abs``,
  ``IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs``, ``IsaacContrib-PickPlace-Locomanipulation-G1-Abs``
  and ``IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs`` to
  :class:`~isaaclab_tasks.contrib.teleop_env.HumanoidTeleopEnv`, which prepares the selected physics
  backend before construction. Code that constructs the environment without that entry point should
  first call :func:`~isaaclab_tasks.contrib.teleop_physics_cfg.prepare_humanoid_teleop_environment`,
  which also applies the preset defaults to a configuration built without task resolution.
* Changed the GR1T2 upper-body action term to
  :class:`~isaaclab_tasks.contrib.pick_place.gr1t2_kitless_action_cfg.GR1T2KitlessPinkInverseKinematicsActionCfg`.
  With Isaac Sim PhysX it behaves as the previous Pink term.
* Changed fixed-base G1 to re-render three times after each reset (``num_rerenders_on_reset = 3``),
  which adds render work per reset when cameras are rendered.
* Changed the G1 hand contact-sensor paths to name the ``left_hand``/``right_hand`` parent of the hand
  links in the Isaac 6.2 ``g1.usd``. Newton matches whole body paths, so it needs that parent; Isaac Sim
  PhysX and OvPhysX resolve the same links by name as before.
