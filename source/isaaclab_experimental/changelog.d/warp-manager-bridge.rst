Added
^^^^^

* Added :mod:`isaaclab_experimental.envs.frontend`, a small runtime selector
  used by ``--frontend {torch,warp}`` to choose how a task is constructed.

  * ``torch`` (default) dispatches via :func:`gym.make` unchanged.
  * ``warp`` adapts a stable manager-based cfg in place and constructs
    :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`. The adapter
    does three things, all hard-failing on incompatibility:

    1. Requires :class:`~isaaclab_newton.physics.NewtonCfg` as the active
       physics — the user is responsible for selecting it via
       ``presets=newton``; no Hydra-arg mutation happens behind the scenes.
    2. Promotes every stable :class:`~isaaclab.managers.SceneEntityCfg`
       embedded under term ``params`` to
       :class:`isaaclab_experimental.managers.SceneEntityCfg` via the new
       :meth:`~isaaclab_experimental.managers.SceneEntityCfg.from_stable`
       classmethod (no ``__class__`` reassignment).
    3. Swaps every stable ``term.func`` *and* ``term.class_type`` (one pass,
       handles observations / events / rewards / terminations / commands /
       curriculum / actions) with its same-named warp twin from
       ``isaaclab_tasks_experimental.<task>.mdp`` or
       :mod:`isaaclab_experimental.envs.mdp`. Any missing twin raises
       :class:`FrontendIncompatibleError` listing the affected term paths.

  Direct workflows aren't adapted; ``--frontend=warp`` on a direct task
  requires the task to be pre-registered under ``isaaclab_experimental`` /
  ``isaaclab_tasks_experimental`` (e.g. ``*-Direct-Warp-v0``).

* Added :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable`,
  a classmethod that builds a warp scene-entity cfg from a stable one by
  copying every selection field through ``__init__``.

* Added a ``--frontend {torch,warp}`` flag to ``rsl_rl/train.py``. The flag
  selects the runtime via :func:`isaaclab_experimental.envs.frontend.build`;
  ``isaaclab_experimental`` is treated as optional and ``--frontend=torch``
  falls back to ``gym.make`` when it isn't installed.

Fixed
^^^^^

* Fixed a regression in :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`
  introduced when the ``SimulationContext.get_setting`` API was reshaped:
  the warp env now mirrors the stable env and probes
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` instead of
  splitting a string setting that no longer exists.
