* Fixed headless Newton visualizers declared in :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` capturing
  empty frames, for example a video recorded from ``NewtonGLVisualizerCfg(headless=True)``. A headless visualizer
  does not render continuously, so the Newton model was built without its visual shapes.
