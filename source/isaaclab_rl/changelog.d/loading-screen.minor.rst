Added
^^^^^

* Added a startup loading screen to the ``train`` and ``play`` entrypoints. Each run now prints a
  summary of its task, workflow, RL library, physics, renderer, presets, visualizer, device, and
  environment count, followed by a progress bar that hands the console back before training starts.
  The presets row lists any other presets the run selected, such as ``presets=cube``, and reads
  ``none`` when it selected none.
