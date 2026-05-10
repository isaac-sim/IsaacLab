Added
^^^^^

* Added :func:`isaaclab_tasks.utils.preset_cli.setup_cli`, a typed-flag
  layer in front of the existing Hydra-decorator preset flow. Scripts
  call ``setup_cli(parser)`` once and gain ``--physics=NAME``,
  ``--renderer=NAME``, and free-form ``--presets=NAME[,NAME,...]``; the
  flags are translated to a single ``presets=<csv>`` Hydra token, so the
  downstream resolver path is unchanged. ``setup_cli`` validates each
  typed flag against
  :meth:`~isaaclab.utils.preset_registry.PresetRegistry.names_for`
  unioned with the field names found on the selected task's
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` instances, so users can
  define variant alternatives (e.g. ``newton_mjwarp_strict: MjwarpCfg``)
  alongside the canonical-named field without re-decorating the cfg
  class. ``--help`` lists the valid choices per target, scoped to the
  task when ``--task=<X>`` is supplied. Legacy alias inputs
  (``--physics newton`` -> ``newton_mjwarp``;
  ``--physics kamino`` -> ``newton_kamino``) are normalized with a
  :exc:`FutureWarning`.
