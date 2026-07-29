Added
^^^^^

* Added the ``--isaacsim_source`` CLI option, which builds Isaac Sim from a source checkout,
  packages the build as Python wheels, links them into the repository as ``_isaac_sim_wheels``,
  points ``uv`` at that directory through ``find-links`` in ``pyproject.toml``, pins the
  ``isaacsim-local`` extra to the version it built, and re-resolves Isaac Sim from those
  wheels. Run Isaac Lab against the build with ``uv run --extra isaacsim-local``. The pin is
  required because source builds carry pre-release local versions that sort below the published
  release, so an unpinned extra resolves back to the released wheels on ``pypi.nvidia.com``.
* Added a check to ``--isaacsim_source`` that rejects a stale Isaac Sim ``_build`` tree whose
  packaged Kit kernel does not match the Python ABI its wheel is tagged for, instead of letting
  Isaac Sim fail later with ``No module named 'carb._carb'``.
