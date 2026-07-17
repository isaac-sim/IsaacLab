Added
^^^^^

* Added ``--frontend {torch,warp}`` to the shared reinforcement-learning
  training CLI (all supported RL libraries) for selecting the environment
  runtime; default ``torch`` is unchanged.
* Added :mod:`isaaclab_experimental.envs.frontend` runtime selector and
  :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable` used by
  ``--frontend=warp`` to adapt stable cfgs onto the warp runtime.
* Added ``warp_entry_point`` registration support: a stable direct task may
  declare its warp environment class, which ``--frontend=warp`` constructs
  with the stable configuration.
* Added warp adapters for the stable
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` and
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_mass` startup event
  terms, so tasks using them run under ``--frontend=warp`` with randomization
  active.
* Added deterministic warp twin resolution: a stable symbol's twin is looked
  up on the mirrored experimental package tree (``isaaclab.* ↔
  isaaclab_experimental.*``, ``isaaclab_tasks.* ↔
  isaaclab_tasks_experimental.*``); missing twins are collected and reported
  in one failure. No registration is needed.

Changed
^^^^^^^

* Changed :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp` to adapt
  its configuration in ``__init__`` via
  :meth:`~isaaclab_experimental.envs.frontend.WarpFrontend.adapt_cfg`, so
  registered warp task variants can derive from stable configurations instead
  of duplicating them.
* Changed the experimental warp ``RewardManager`` to merge dictionaries
  returned by class-based reward term ``reset()`` into its episode-log extras
  (persistent Warp buffer views, CUDA-graph safe); used to report
  ``Metrics/success_rate`` under ``--frontend warp``.

Fixed
^^^^^

* Fixed :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp` to probe
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` after the
  ``get_setting`` API was reshaped.
