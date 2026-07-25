Changelog
---------

0.1.4 (2026-07-25)
~~~~~~~~~~~~~~~~~~

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


0.1.3 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``,
  ensuring standalone pip installs resolve local checkouts without a package index.


0.1.2 (2026-06-11)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a crash (``AttributeError: 'dict' object has no attribute 'split'``) when
  launching the experimental Warp environments
  (:class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`,
  :class:`~isaaclab_experimental.envs.DirectRLEnvWarp`) with a Kit visualizer
  requested (e.g. ``--visualizer kit``). The environments now resolve the active
  visualizer through :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers`
  and the :attr:`~isaaclab.sim.SimulationContext.is_rendering` property, matching the
  stable environments, instead of parsing the ``/isaaclab/visualizer`` settings node
  (which is a dictionary) as a string.


0.1.1 (2026-06-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the experimental packages eagerly importing backend modules (``pxr``,
  ``omni``, ``carb``, ``isaacsim``, ``scipy``) at import time, which crashed when
  a warp task's env config was loaded before ``SimulationApp`` was launched. The
  ``managers``, ``envs``, ``envs.mdp`` and ``envs.mdp.actions`` packages now use
  ``lazy_export`` with ``.pyi`` stubs, and the MDP term leaf modules guard runtime
  types (``Articulation``, ``InteractiveScene``, ``ContactSensor``, action terms)
  under ``TYPE_CHECKING`` with string ``class_type`` references.


0.1.0 (2026-06-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for :attr:`~isaaclab.managers.EventTermCfg.resample_interval_on_reset` in the
  experimental Warp-first event manager, allowing ``"interval"`` event terms to keep their
  per-environment timer across resets while still firing asynchronously per environment.

Fixed
^^^^^

* Fixed the Warp gravity kernels behind
  :func:`~isaaclab_experimental.envs.mdp.projected_gravity` and
  :func:`~isaaclab_experimental.envs.mdp.flat_orientation_l2` to read per-env
  gravity and normalize it, instead of reading env 0's vector. Per-env gravity
  randomization is now respected by the observation and the flat-orientation
  reward on the Newton backend.


0.0.5 (2026-05-18)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :mod:`isaaclab_experimental.utils` package exports so its utility
  modules appear in API documentation.


0.0.4 (2026-05-12)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Pre-create renderer backends in
  :class:`~isaaclab_experimental.envs.ManagerBasedEnvWarp` and
  :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` by invoking
  :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers` after scene
  construction so that renderer backend creation order is deterministic and
  front-loaded before the first
  :meth:`~isaaclab.sim.SimulationContext.reset`.


0.0.3 (2026-04-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the Warp-graphable MDP terms and the Warp inhand-manipulation env to read
  asset/sensor data via the explicit :attr:`~isaaclab.utils.warp.ProxyArray.warp`
  accessor when the value flows into a ``wp.launch`` call (or a sim-write helper that
  forwards to one). Affected modules:
  :mod:`isaaclab_experimental.envs.mdp.observations`,
  :mod:`isaaclab_experimental.envs.mdp.rewards`,
  :mod:`isaaclab_experimental.envs.mdp.terminations`,
  :mod:`isaaclab_experimental.envs.mdp.events`,
  :mod:`isaaclab_experimental.envs.mdp.actions.joint_actions`, and
  :mod:`isaaclab_tasks_experimental.direct.inhand_manipulation.inhand_manipulation_warp_env`.
  The previous code relied on ``ProxyArray``'s ``__cuda_array_interface__`` bridge,
  which works but is not explicit. No behavior change.
* Replaced ``wp.to_torch(asset.data.joint_pos).shape[1]`` in
  :class:`~isaaclab_experimental.managers.ObservationManager` with
  ``asset.data.joint_pos.shape[1]`` — :class:`~isaaclab.utils.warp.ProxyArray` forwards
  ``shape``, so the round-trip through ``wp.to_torch`` is no longer needed.


0.0.2 (2026-03-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` not being recognized by
  RL library wrappers (e.g. :class:`~isaaclab_rl.rl_games.RlGamesVecEnvWrapper`) that
  check for :class:`~isaaclab.envs.DirectRLEnv` via ``isinstance``. Changed base class
  from :class:`gym.Env` to :class:`~isaaclab.envs.DirectRLEnv`; all methods are
  overridden so behavior is unchanged.


0.0.1 (2026-01-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_experimental`` package.
