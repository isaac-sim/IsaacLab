Added
^^^^^

* Added :class:`~isaaclab_tasks.utils.TaskVariantCfg` metadata so tasks can
  declare compatible domain presets for each registered agent config. Task
  help and ``--list_variants`` now expose registered agents and compatibility,
  and incompatible selections fail before simulator launch.
* Added preset-to-agent compatibility for :obj:`Isaac-Cartpole-Camera`,
  :obj:`IsaacContrib-Cartpole-Showcase-Direct`, and
  :obj:`IsaacContrib-Cartpole-Camera-Showcase-Direct`.
