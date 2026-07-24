Changed
^^^^^^^

* **Breaking:** Changed the package layout to mirror :mod:`isaaclab_tasks.core`
  (``isaaclab_tasks_experimental.core.<task>``), replacing the previous
  ``manager_based``/``direct`` split. Update imports of the old paths to the
  ``core`` equivalents (e.g. ``isaaclab_tasks_experimental.core.cartpole.mdp``).
* Changed the direct Warp env modules and classes to the ``<task>_warp_env`` /
  ``<Name>WarpEnv`` convention so ``--frontend warp`` resolves them by name:
  renamed ``ant_env_warp`` to ``ant_warp_env`` and ``InHandManipulationWarpEnv``
  (module ``inhand_manipulation_warp_env``) to ``ReorientDirectWarpEnv``
  (module ``reorient_warp_env``).

Removed
^^^^^^^

* **Breaking:** Removed all manager-based ``*-Warp-v0`` task registrations —
  ``Isaac-Cartpole-Warp-v0``, ``Isaac-Humanoid-Warp-v0``, ``Isaac-Ant-Warp-v0``,
  ``Isaac-Reach-Franka-Warp-v0``, ``Isaac-Reach-Franka-Warp-Play-v0``, and the
  velocity variants (``Isaac-Velocity-Flat-<Robot>-Warp-v0`` and their
  ``-Warp-Play-v0`` forms) — together with their environment configurations.
  Run the stable task ids with ``--frontend warp`` and
  ``presets=newton_mjwarp`` instead, e.g.
  ``--task Isaac-Velocity-Flat-AnymalD --frontend warp presets=newton_mjwarp``.
* **Breaking:** Removed the duplicated direct warp task registrations
  ``Isaac-Cartpole-Direct-Warp-v0``, ``Isaac-Ant-Direct-Warp-v0``,
  ``Isaac-Humanoid-Direct-Warp-v0``, and
  ``Isaac-Reorient-Cube-Allegro-Direct-Warp-v0`` together with their
  duplicated environment configurations; the frontend resolves their Warp
  environment classes by the mirrored naming convention, with
  ``warp_entry_point`` available as an optional override. Run the stable task
  ids with ``--frontend warp`` and ``presets=newton_mjwarp`` instead, e.g.
  ``--task Isaac-Cartpole-Direct --frontend warp presets=newton_mjwarp``.
* Removed the unregistered rough velocity warp configurations; rough-terrain
  warp tasks remain unsupported until :class:`~isaaclab.terrains.TerrainImporter`
  gains Warp APIs.

Fixed
^^^^^

* Fixed the direct Warp Cartpole task to match the stable task's observations,
  reset ranges, termination condition, reward scaling, and scene configuration.
* Fixed the Warp Cartpole ``survival_success_rate`` twin to report the
  ``Metrics/success_rate`` value on-device instead of silently dropping the
  metric.
* Fixed ``Isaac-Humanoid-Direct`` under ``--frontend warp`` failing at env
  construction: the Warp locomotion env now resolves the per-physics-backend
  ``joint_gears`` dict (selecting the Newton ordering) like the stable
  ``LocomotionDirectEnv``, instead of passing the dict straight to ``wp.array``.
