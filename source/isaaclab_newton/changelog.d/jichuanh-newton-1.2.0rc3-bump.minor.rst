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
* ``mjwarp`` is bumped from ``3.8.0.1`` to ``3.8.0.2`` transitively via
  ``newton[sim]``; no IsaacLab-side pin change required since
  :pr:`5566` dropped IsaacLab's explicit ``mujoco`` / ``mujoco-warp``
  pins.
