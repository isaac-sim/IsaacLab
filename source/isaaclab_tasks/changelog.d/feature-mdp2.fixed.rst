* Fixed the task auto-import skipping packages whose names start with a blacklisted name, such as
  :mod:`isaaclab_tasks.mdp2` for ``.mdp``. Blacklisted names now match whole package names.
* Fixed :func:`~isaaclab_tasks.utils.hydra.resolve_presets` reassigning unchanged tuples, which failed on frozen
  dataclasses in the configuration tree.
