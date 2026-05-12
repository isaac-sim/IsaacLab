Added
^^^^^

* Added :class:`isaaclab_tasks.utils.preset_target.PresetTarget` -- closed enum
  of typed CLI preset categories (``PHYSICS``, ``RENDERER``, ``DOMAIN``). Each
  member carries the typed-flag label, the cfg base classes whose subclass
  instances belong in that bucket, and a per-target legacy-alias table. The
  resolver in :mod:`isaaclab_tasks.utils.hydra` reads
  :meth:`~isaaclab_tasks.utils.preset_target.PresetTarget.all_legacy_aliases`
  as the single source of truth for legacy alias rewrites.
* Added :func:`isaaclab_tasks.utils.preset_cli.setup_preset_cli` -- a typed-flag
  argparse layer over the existing ``presets=<csv>`` Hydra-decorator preset
  flow. Scripts call ``setup_preset_cli(parser)`` once and gain
  ``--physics=NAME``, ``--renderer=NAME``, and free-form
  ``--presets=NAME[,NAME,...]``. The function returns ``(args, hydra_argv)``
  without mutating ``sys.argv``; the caller assigns
  ``sys.argv = [sys.argv[0]] + hydra_argv`` when ready, so any argv-aware logic
  (e.g., an ``--external_callback`` hook that re-reads ``sys.argv``) sees the
  user's original command line first. Hydra's existing
  :func:`~isaaclab_tasks.utils.hydra.resolve_presets` consumes the folded
  token unchanged. The discoverability win: when ``--task=X`` is given
  alongside ``--help``, the help text lists the
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` variants actually present
  in ``X``'s env_cfg (walked via
  :func:`~isaaclab_tasks.utils.hydra.collect_presets`) and buckets them by
  ``isinstance`` against
  :attr:`~isaaclab_tasks.utils.preset_target.PresetTarget.base_classes` so
  typed flags list only their own kind. Unknown names pass through verbatim;
  hydra's existing
  :func:`~isaaclab_tasks.utils.hydra._format_unknown_presets_error` produces
  the rich error at resolve time as before.

Changed
^^^^^^^

* Changed :mod:`isaaclab_tasks.utils.hydra` to source legacy preset aliases
  from
  :meth:`~isaaclab_tasks.utils.preset_target.PresetTarget.all_legacy_aliases`
  instead of a local literal dict; per-target alias tables now live on the
  enum members.
