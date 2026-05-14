Added
^^^^^

* Added :class:`isaaclab_tasks.utils.preset_target.PresetTarget` -- closed enum
  of typed preset categories (``PHYSICS``, ``RENDERER``, ``DOMAIN``).
* Added :func:`isaaclab_tasks.utils.preset_cli.setup_preset_cli` -- a typed
  selection layer over the ``presets=<csv>`` Hydra-decorator preset flow.
  Recognizes three Hydra-style tokens (``physics=NAME``, ``renderer=NAME``,
  ``presets=NAME[,...]``) and folds them into the existing token. When
  ``--task=X`` is given alongside ``--help``, lists the
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` variants present in the
  task's env_cfg, bucketed by typed target.

Changed
^^^^^^^

* Changed :mod:`isaaclab_tasks.utils.hydra` to source legacy preset aliases
  from :meth:`~isaaclab_tasks.utils.preset_target.PresetTarget.all_legacy_aliases`
  instead of a local literal dict.
