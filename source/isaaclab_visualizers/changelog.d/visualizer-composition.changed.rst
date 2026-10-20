* Separated backend options into ``KitBackendCfg``, ``NewtonGLBackendCfg``, ``NewtonRTXBackendCfg``,
  ``RerunBackendCfg``, and ``ViserBackendCfg``. Constructors received shared ``VisualizerCfg``
  declarations and read implementation options from ``cfg.backend``. Kept the previous
  ``*VisualizerCfg`` names as deprecated configuration-only adapters; move shared camera/window
  settings to ``VisualizerCfg`` and backend-specific options to its ``backend``.
* Enabled Newton GL particle display by default. Set ``NewtonGLBackendCfg(show_particles=False)``
  to hide particles; hidden MPM particles no longer updated their display buffers.
