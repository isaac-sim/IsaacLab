* **Breaking:** The zero and random agents no longer open the Newton GL visualizer by default. Pass
  ``--visualizer newton_gl`` to open it.
* **Breaking:** ``--video`` takes an optional source, ``--video [SOURCE]``, and ``--visualizer`` still decides
  which visualizers open a window:

  * ``--video`` (``--video viz``): the first capture-capable visualizer ``--visualizer`` selects, else a headless
    ``newton_gl``, also when only streaming visualizers such as ``viser`` or ``rerun`` are selected.
  * ``--video viz:<type>`` (``kit``, ``newton_gl``, ``newton_rtx``): the selected visualizer of that type, else an
    extra headless one, e.g. ``--video viz:newton_gl --visualizer viser``.
  * ``--video sensor:<name>[:<channel>]``: that scene sensor; no visualizer is added.

  A Hydra override is never taken as the source: ``--video presets=newton_mjwarp`` records from ``viz`` and
  applies the preset. ``--video`` without ``--visualizer`` now records from a headless ``newton_gl`` instead of a
  headless Kit; pass ``--video viz:kit`` for the previous behavior.
* **Breaking:** :func:`~isaaclab_rl.entrypoints.common.pre_launch_video_config`, called before
  :func:`~isaaclab.app.launch_simulation`, adds a recorder for the ``--video`` source unless the environment config
  declares recorders, and no longer selects ``--visualizer kit`` or sets ``headless``; the launch resolves the
  source. :func:`~isaaclab_rl.entrypoints.common.apply_video_recording`, called inside the launch, only applies the
  output directory, ``--video_length`` and ``--video_interval``. The zero and random agents now configure
  recording after the launch.
* :func:`~isaaclab_rl.entrypoints.common.enable_cameras_for_video` only enables cameras for
  ``--capture_env_sensors``; the launch enables the rendering a video source needs.
* The ``video`` field of the :mod:`isaaclab_rl.entrypoints.api` requests takes a ``--video`` source string as
  well as a bool.
* ``get_pretrained_checkpoint_backend_names`` ignores the renderers of visualizer configs, so a published
  checkpoint lookup with a visualizer selected, e.g. ``--visualizer newton_gl`` or ``--visualizer kit``, no longer
  asks for a ``newton`` or ``rtx`` render-backend checkpoint.
