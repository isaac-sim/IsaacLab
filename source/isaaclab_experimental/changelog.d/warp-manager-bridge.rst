Added
^^^^^

* Added :class:`~isaaclab_experimental.envs.warp_frontend.WarpFrontend`, a
  runtime adapter that lets any stable manager-based RL task config run on the
  experimental warp manager runtime (:class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`)
  without a parallel ``-Warp-v0`` registration. The adapter resolves
  :class:`~isaaclab_physx.preset.PresetCfg` to its ``newton`` field, swaps
  ``term.func`` references to same-named warp twins discovered in the warp
  ``mdp`` modules (skipping stable re-exports), promotes ``SceneEntityCfg``
  instances in-place to the warp variant, and reports any missing twins
  before the env is built.

* Added a ``--manager={stable,warp}`` flag to ``rsl_rl/train.py``. When set
  to ``warp`` the script auto-injects ``presets=newton`` (so Hydra picks the
  Newton physics preset before the adapter runs) and dispatches the env
  through ``WarpFrontend`` instead of ``gym.make``.

Fixed
^^^^^

* Fixed a regression in :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`
  introduced when the ``SimulationContext.get_setting`` API was reshaped:
  the warp env now mirrors the stable env and probes
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` instead of
  splitting a string setting that no longer exists.
