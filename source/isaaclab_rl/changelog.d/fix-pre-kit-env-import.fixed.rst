* Fixed Isaac Sim crashing at startup in OpenBLAS's at-fork handler when ``moviepy`` is installed: the
  reinforcement learning entry points imported the environment runtime, and with it the video recorder's
  ``moviepy`` and SciPy, before the launch.
