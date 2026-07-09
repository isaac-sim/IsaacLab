Fixed
^^^^^

* Fixed the headless RTX video pump so it still updates Kit on demand when a frame is requested,
  after :attr:`~isaaclab.sim.SimulationContext.is_rendering` stopped reporting offscreen rendering
  as continuous rendering. Offscreen frames are now pumped only when requested, not every step.
* Fixed physics corruption when recording video with ``--video --device cpu``: the Kit
  ``app.update()`` inside :class:`~isaaclab_physx.video_recording.IsaacsimKitPerspectiveVideo`
  now guards ``/app/player/playSimulations`` so physics is not advanced mid-render.
