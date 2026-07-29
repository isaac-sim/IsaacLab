Changed
^^^^^^^

* **Breaking:** Removed ``eye`` and ``lookat`` fields from the Kit perspective video recorder
  config.  The Kit perspective recorder no longer repositions the viewport camera; camera
  placement is the sole responsibility of :class:`~isaaclab_visualizers.kit.KitVisualizer`.

* Added :meth:`~isaaclab_physx.physics.PhysxManager.video_capture_backend` classmethod
  (returns ``"kit"``). The headless video pump is now registered via
  :meth:`~isaaclab.sim.SimulationContext.add_render_callback` in
  :meth:`~isaaclab_physx.physics.PhysxManager.initialize` instead of the
  deleted ``recording_hooks`` module.
