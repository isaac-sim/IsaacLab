Changed
^^^^^^^

* Switched the Newton install to ``newton[sim]`` so that ``mujoco`` and
  ``mujoco-warp`` are pulled in transitively via Newton's ``[sim]`` extra.
  The explicit ``mujoco==3.8.0`` and ``mujoco-warp==3.8.0.1`` pins were
  removed from :mod:`isaaclab_newton` — Newton is now the single source of
  truth for those versions.
