Changed
^^^^^^^

* Removed the ``replicate_physics`` guards that raised ``RuntimeError`` on USD-level randomization
  (scale, visual color, and visual texture) in :class:`~isaaclab.managers.EventManager` and the
  ``randomize_*`` event functions. The Isaac Lab cloner now replicates per-object when environments
  differ, so prestartup USD randomization is preserved per-environment with ``replicate_physics=True``.
