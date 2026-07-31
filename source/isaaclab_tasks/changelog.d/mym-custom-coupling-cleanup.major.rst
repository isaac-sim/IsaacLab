Changed
^^^^^^^

* **Breaking:** Removed the ``newton_mjwarp_vbd`` preset from
  ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Cloth-Franka``. Both tasks now
  default to ``newton_mjwarp_vbd_proxy``, which uses proxy coupling. Select
  ``presets=newton_mjwarp_vbd_proxy`` explicitly, or import
  :mod:`isaaclab_contrib.custom_coupling.tasks` and use its
  ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling`` task, which adds back a
  ``newton_mjwarp_vbd`` preset for the manual coupler.
