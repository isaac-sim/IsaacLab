Changed
^^^^^^^

* **Breaking:** Replaced the ``SceneDataRequirement`` dataclass and its resolution helpers with two
  public boolean attributes on :class:`~isaaclab.sim.SimulationContext`,
  ``requires_usd_stage`` and ``requires_newton_model``, and with the
  ``REQUIRES_STAGE_AND_MODEL`` mapping exported from
  :mod:`isaaclab.scene_data`. Read a requirement with ``sim.requires_newton_model`` instead of
  ``sim.get_scene_data_requirements().requires_newton_model``, and publish one by OR-ing the flags
  directly, for example::

      requires_stage, requires_model = REQUIRES_STAGE_AND_MODEL["newton_warp"]
      sim.requires_usd_stage |= requires_stage
      sim.requires_newton_model |= requires_model

Removed
^^^^^^^

* **Breaking:** Removed the ``isaaclab.physics.scene_data_requirements`` module, including
  ``SceneDataRequirement``, ``resolve_scene_data_requirements``, ``aggregate_requirements``, and the
  per-type requirement lookups, along with ``SimulationContext.get_scene_data_requirements`` and
  ``SimulationContext.update_scene_data_requirements``. Use the attributes and mapping described
  above instead.
