Changed
^^^^^^^

* Bumped Newton pin to ``v1.2.0rc3``. Pulls in IsaacLab-relevant fixes
  from `newton-physics/newton#2651
  <https://github.com/newton-physics/newton/pull/2651>`_ (MPR/GJK no
  longer assumes convex hulls are centered around the origin),
  `newton-physics/newton#2703
  <https://github.com/newton-physics/newton/pull/2703>`_ (Kamino FK
  solver performance), `newton-physics/newton#2721
  <https://github.com/newton-physics/newton/pull/2721>`_ (HDR color
  output for tiled camera sensors), plus SolverMuJoCo fixes for planar
  meshes, contact-anchor computation, and distance conversion.
* The ``newton[sim]`` transitive deps now resolve to ``mjwarp==3.8.0.2``
  (was ``3.8.0.1``). IsaacLab no longer pins ``mujoco`` / ``mujoco-warp``
  explicitly (see :pr:`5566`), so these flow in through ``newton[sim]``
  with no IsaacLab-side pin change.
* The Newton pin is mirrored across :mod:`isaaclab_newton`,
  :mod:`isaaclab_visualizers` (3×), :mod:`isaaclab_physx` (``[newton]``
  extra), and ``tools/wheel_builder/res/python_packages.toml``.
