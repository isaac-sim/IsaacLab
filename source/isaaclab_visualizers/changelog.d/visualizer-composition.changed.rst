* **Breaking:** Separated backend options from common ``VisualizerCfg`` settings. Updated visualizer
  constructors to receive ``VisualizerCfg(backend=NewtonGLVisualizerCfg(), ...)`` (or another backend).
  Moved view, window, camera, and environment-selection arguments to the outer configuration.
* Enabled Newton GL particle display by default. Set ``NewtonGLVisualizerCfg(show_particles=False)``
  to hide particles; hidden MPM particles no longer updated their display buffers.
