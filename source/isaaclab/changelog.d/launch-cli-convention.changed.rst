* **Breaking:** Without ``--visualizer``, :func:`~isaaclab.app.launch_simulation` runs no visualizer, even if
  :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` lists some. ``--visualizer`` selects which visualizers
  run and the configured ones only supply the settings of the selected types: each selected type uses the
  configured visualizer of that type, or else its default config. Configured visualizers no longer start Kit,
  OVRTX, or cameras unless selected. To keep running a visualizer your config lists, pass
  ``--visualizer <type>`` (or ``launch_simulation(cfg, {"visualizer": "<type>"})``). A
  :class:`~isaaclab.sim.SimulationContext` built without a launch still runs the configured visualizers as given.
* **Breaking:** Tasks, ``run_cartpole_rl_env.py``, ``lift_franka_soft.py`` and ``check_keyboard.py`` no longer
  open a visualizer by default. Pass ``--visualizer`` (for example ``--visualizer kit`` or
  ``--visualizer newton_gl``) to open one. Demos, examples and visualizer tutorials keep their default visualizer
  through ``parser.set_defaults(visualizer=[...])``; ``--visualizer`` replaces it.
* **Breaking:** ``--visualizer`` accepts only lower-case type names; ``--visualizer Kit`` is rejected with the list
  of valid names.
* **Breaking:** :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.source` takes ``viz``,
  ``viz:<type>``, ``viz:<type>:streaming_view`` or ``sensor:<name>[:<channel>]`` and defaults to ``"viz"``.
  :func:`~isaaclab.app.launch_simulation` resolves the recorder sources once: ``viz`` records from the first
  capture-capable visualizer ``--visualizer`` selects, else from ``newton_gl``, and a ``viz:<type>`` that
  ``--visualizer`` does not select is added to :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` headless,
  from the configured visualizer of that type or its default config, only for the recording. The source is
  rewritten to the concrete ``viz:<type>``. Recording from ``viz:rerun`` or ``viz:viser`` raises a
  :class:`ValueError`, as streaming visualizers have no frame capture. ``visualizer`` is accepted as the
  long form of the ``viz`` prefix; the ``newton`` type is a deprecated alias of ``newton_gl``.
* The benchmark play entry points take ``--video [SOURCE]`` and ``--video_interval`` like the training entry
  points.
* :meth:`~isaaclab.sim.SimulationContext.can_render_rgb_array` counts headless visualizers, so a Newton model
  imports its visual shapes when only a headless visualizer, e.g. one a video records from, draws it.
* **Breaking:** Isaac Sim / Kit runs windowed only when ``--visualizer`` selects ``kit``, and never with
  ``HEADLESS=1`` or livestreaming; a Kit visualizer only a video records from runs headless.
* ``/isaaclab/visualizer/types`` holds the launch's visualizers on every launch: a comma-separated list of the
  selected types and the types video recorders add, empty when there are none.
