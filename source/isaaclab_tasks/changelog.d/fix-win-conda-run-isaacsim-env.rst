Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.utils.sim_launcher.launch_simulation` failing with
  "Isaac Sim is not installed or not found on PYTHONPATH" on Windows when the script
  is run via ``conda run`` (e.g. from CI automation).  ``conda run`` does not fire
  ``activate.d`` hooks, so ``setup_conda_env.bat`` was never sourced into the training
  script process.  The new :func:`~isaaclab_tasks.utils.sim_launcher._try_setup_isaacsim_env_windows`
  helper now actively applies ``setup_conda_env.bat``'s environment changes (PYTHONPATH,
  PATH, and ``os.add_dll_directory`` for Python 3.8+ DLL loading) directly in the
  running process before erroring out.  Also updated ``isaaclab.bat`` to use a
  belt-and-suspenders ``for /f cmd /c "call ... && set"`` approach alongside the
  existing ``call``, handling the case where ``setup_conda_env.bat`` uses
  ``setlocal``/``endlocal`` internally.  Addresses NVBug 5984996.
