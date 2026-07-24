Added
^^^^^

* Added :mod:`isaaclab_experimental.envs.frontend` runtime selector and
  :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable` used by
  ``--frontend=warp`` to adapt stable cfgs onto the warp runtime.
* Added name-based resolution of direct-task warp env classes: ``--frontend=warp``
  resolves a stable direct task's ``<task>_direct_env:<Name>Env`` to the mirrored
  ``<task>_warp_env:<Name>WarpEnv`` and constructs it with the stable cfg. No
  parallel registration is needed; a ``warp_entry_point`` kwarg is honored as an
  optional override for classes that cannot follow the convention. The resolved
  twin is verified against the task's cfg (via the warp env's ``cfg`` annotation),
  so a stable env class shared by several tasks — e.g. ``ReorientDirectEnv``,
  which serves both the Allegro and Shadow hands — never silently runs one
  variant's warp env for another; a mismatch is a hard error.
* Added observation-noise conversion to the warp frontend: stable noise cfgs
  (constant/uniform/gaussian) swap to their warp-native twins during cfg
  adaptation; cfgs without a twin (class-based noise models, customized or
  tensor-valued parameters) are a hard error instead of being silently
  ignored. The Warp observation manager rejects non-warp noise cfgs and
  plumbs the shared per-env RNG state to function-style noise kernels.
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
