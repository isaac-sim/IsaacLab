Added
^^^^^

* Added the ``--isaacsim_source`` CLI option, which builds Isaac Sim from a source checkout,
  packages the build as Python wheels, links them into the repository as ``_isaac_sim_wheels``,
  and re-resolves Isaac Sim from those wheels. Run Isaac Lab against the build with
  ``uv run --extra isaacsim-local``.
