Changed
^^^^^^^

* **Breaking:** Changed the manager-based velocity ``*-Warp-v0`` task variants
  to reuse the stable flat configurations, disabling only the randomization
  events that have no warp twins yet. The variants now require selecting the
  Newton solver on the CLI via ``presets=newton_mjwarp`` instead of
  hard-coding it in the configuration.

Removed
^^^^^^^

* **Breaking:** Removed the duplicated manager-based warp task registrations
  ``Isaac-Cartpole-Warp-v0``, ``Isaac-Humanoid-Warp-v0``, ``Isaac-Ant-Warp-v0``,
  ``Isaac-Reach-Franka-Warp-v0``, and ``Isaac-Reach-Franka-Warp-Play-v0``
  together with their duplicated environment configurations. Run the stable
  task ids with ``--frontend warp`` and ``presets=newton_mjwarp`` instead,
  e.g. ``--task Isaac-Cartpole --frontend warp presets=newton_mjwarp``.
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
