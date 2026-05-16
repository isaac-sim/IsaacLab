Fixed
^^^^^

* Excluded the broken ``numpy 2.3.5`` release from every install path that pulls
  numpy. ``numpy 2.3.5``'s vendored OpenBLAS
  (``libscipy_openblas64_-fdde5778.so``) registers a buggy ``pthread_atfork``
  handler that crashes Kit's ``libomni.platforminfo`` ``fork()`` during
  ``SimulationApp`` startup. The exclusion is declared at every site:

  * Each ``source/<pkg>/setup.py`` that depends on numpy directly or
    transitively (``isaaclab``, ``isaaclab_tasks``, ``isaaclab_rl``,
    ``isaaclab_visualizers``, ``isaaclab_teleop``, ``isaaclab_mimic``).
  * The ``pin-pink`` force-reinstall in
    :meth:`isaaclab.cli.commands.install._ensure_pink_ik_dependencies_installed`.
  * The ARM ``setuptools wheel numpy`` pre-install in
    :meth:`isaaclab.cli.commands.install._maybe_preinstall_arm_nlopt`.
  * The ARM nlopt prep step in ``docker/Dockerfile.base``.
  * The ``nvidia-curobo`` install in ``docker/Dockerfile.curobo``.

  See numpy/numpy#30092 and OMPE-92261.
