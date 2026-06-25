Added
^^^^^

* Extended :class:`~isaaclab.envs.mdp.randomize_visual_color` to the kit-less Newton-Warp and OVRTX
  renderers in addition to Kit/Replicator, selecting the backend from the scene's renderer. Colors are
  applied per-prim and per-environment and honor an ``env_ids`` subset on reset.
* Added pre-``PHYSICS_READY`` event-term setup via
  :meth:`~isaaclab.managers.EventManager.initialize_pre_physics_ready_terms`, letting an event term
  author USD before the first physics step.
* Added :mod:`isaaclab.utils.visual_color` with the shared per-environment target selection used by the
  visual-color writer backends.
