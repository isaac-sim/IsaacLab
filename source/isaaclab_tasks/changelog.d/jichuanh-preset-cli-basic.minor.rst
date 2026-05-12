Added
^^^^^

* Added :func:`isaaclab_tasks.utils.preset_cli.setup_cli` -- a typed-flag
  argparse layer over the existing ``presets=<csv>`` Hydra-decorator
  preset flow. Scripts call ``setup_cli(parser)`` once and gain
  ``--physics=NAME``, ``--renderer=NAME``, and free-form
  ``--presets=NAME[,NAME,...]``. Flag values are folded into a single
  ``presets=<csv>`` token in ``sys.argv``; the existing resolver
  (:func:`isaaclab_tasks.utils.hydra.resolve_presets`) consumes it
  unchanged. The new value is discoverability: ``--help`` lists the
  canonical names registered with
  :func:`isaaclab.utils.preset_registry.register` plus the
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` field names declared on
  the currently selected task (``--task=<X> --help``). Unknown names
  emit a stderr hint and pass through; hydra's existing
  :func:`~isaaclab_tasks.utils.hydra._format_unknown_presets_error`
  produces the rich error at resolve time as before.
