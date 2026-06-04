Changelog
---------

0.0.6 (2026-06-04)
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
