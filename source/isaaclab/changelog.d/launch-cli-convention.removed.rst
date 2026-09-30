* **Breaking:** Removed ``--visualizer none``. Omit ``--visualizer`` to run without visualizers.
* **Breaking:** Removed the ``headless`` launcher argument of :func:`~isaaclab.app.launch_simulation`, which the
  video helpers set internally; the ``--visualizer`` selection, ``HEADLESS=1`` and livestreaming decide whether
  Kit opens a window.
* **Breaking:** Removed the ``visualizer_intent`` launcher argument of :func:`~isaaclab.app.launch_simulation`
  and the ``kit_visualizer`` launcher argument it wrote. Pass ``visualizer="kit"`` to request the Kit
  visualizer.
