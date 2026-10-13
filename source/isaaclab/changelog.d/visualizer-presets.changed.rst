* **Breaking:** Removed ``scan`` and ``Scan``; runtime selection was owned by ``launch_simulation``.
  Read selected physics, renderer, and visualizer configs inside the launch context. Pass the same
  ``SimulationCfg`` to launch and simulation construction instead of transferring viewer names through settings.
* Ran concrete ``SimulationCfg.visualizer_cfgs`` as declared when no CLI selection was supplied.
  Used ``--viz none`` or an empty configuration list to disable viewers.
