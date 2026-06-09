Fixed
^^^^^

* Fixed :func:`~isaaclab.controllers.utils.import_lula` failing to locate the ``lula`` module on
  pip-based Isaac Sim installs by extending :func:`~isaaclab.controllers.utils.find_lula_prebundle_dir`
  to also search the ``extscache/<name>-<version>`` and ``extsDeprecated/<name>`` layouts used by pip
  installs (the Isaac Sim 6.0 pip packages ship ``isaacsim.robot_motion.lula`` under
  ``extsDeprecated``, where the Kit resolver reports it as unavailable), in addition to the
  ``exts/<name>`` layout used by binary installs. :func:`~isaaclab.controllers.utils.import_lula` now
  adds the prebundle to ``sys.path`` before attempting to enable the Kit extension, avoiding a spurious
  ``Failed to resolve extension dependencies`` error logged when Kit tries to enable the deprecated
  ``isaacsim.robot_motion.lula`` extension even though ``lula`` itself loads correctly. When ``lula``
  still cannot be found, a clear, actionable :class:`ModuleNotFoundError` is raised instead of the bare
  import error.
