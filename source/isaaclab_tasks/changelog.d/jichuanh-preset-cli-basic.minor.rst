Added
^^^^^

* Added :func:`isaaclab_tasks.utils.preset_cli.setup_cli` -- a typed-flag
  argparse layer over the existing ``presets=<csv>`` Hydra-decorator
  preset flow. Scripts call ``setup_cli(parser)`` once and gain
  ``--physics=NAME``, ``--renderer=NAME``, and free-form
  ``--presets=NAME[,NAME,...]``. Flag values are folded into a single
  ``presets=<csv>`` token in ``sys.argv``; the existing resolver
  (:func:`isaaclab_tasks.utils.hydra.resolve_presets`) consumes it
  unchanged. The new value is discoverability: when ``--task=X`` is
  given alongside ``--help``, the help text lists the
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` variants actually
  present in ``X``'s env_cfg (walked via
  :func:`~isaaclab_tasks.utils.hydra.collect_presets`) and buckets them
  by target via :class:`~isaaclab.utils.preset_registry.PresetRegistry`
  so typed flags list only their own kind. Unknown names pass through
  verbatim; hydra's existing
  :func:`~isaaclab_tasks.utils.hydra._format_unknown_presets_error`
  produces the rich error at resolve time as before.
