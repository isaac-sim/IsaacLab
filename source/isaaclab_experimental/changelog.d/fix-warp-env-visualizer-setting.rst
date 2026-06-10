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
