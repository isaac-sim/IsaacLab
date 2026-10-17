* **Breaking:** Removed ``scan`` and ``Scan``; runtime selection was shared by ``validate_launch`` and ``launch_simulation``.
  Read selected physics, renderer, and visualizer configs inside the launch context. Pass the same
  ``SimulationCfg`` to launch and simulation construction instead of transferring viewer names through settings.
* Ran concrete ``SimulationCfg.visualizer_cfgs`` as declared when no CLI selection was supplied.
  Used ``--viz none`` or an empty configuration list to disable viewers.
* **Breaking:** Restricted the teapot-fill demo to Newton GL and removed its Kit/RTX viewer setup.
  Omit ``--visualizer`` or use ``--visualizer newton_gl``; use the surface-reconstruction tutorial for RTX rendering.
