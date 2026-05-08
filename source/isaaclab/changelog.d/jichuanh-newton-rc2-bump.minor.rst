Changed
^^^^^^^

* Bumped Newton pin to ``v1.2.0rc2``. Pulls in IsaacLab-relevant fixes from
  `newton-physics/newton#2678 <https://github.com/newton-physics/newton/pull/2678>`_
  and `newton-physics/newton#2720
  <https://github.com/newton-physics/newton/pull/2720>`_ (``SolverKamino``
  reset under ``world_mask``), the upstream tendon-scoping fix from
  `newton-physics/newton#2659
  <https://github.com/newton-physics/newton/pull/2659>`_ ("Scope USD
  custom-frequency parsing"), and a VRAM-leak fix on example reset
  (`newton-physics/newton#2710
  <https://github.com/newton-physics/newton/pull/2710>`_).
* Newton ``v1.2.0rc2`` requires ``warp-lang==1.13.0``, ``mujoco==3.8.0``,
  and ``mujoco-warp==3.8.0.1``. ``warp-lang``/``mujoco``/``mujoco-warp``
  pins live in :mod:`isaaclab` and ``tools/wheel_builder/res/python_packages.toml``;
  the Newton pin is mirrored across :mod:`isaaclab_newton`,
  :mod:`isaaclab_visualizers` (3×), :mod:`isaaclab_physx` (``[newton]``
  extra), and the wheel-builder TOML.
* Updated ``wp.math.transform_to_matrix`` to ``wp.transform_to_matrix`` in
  :mod:`~isaaclab_newton.physics.newton_manager` and
  :mod:`~isaaclab_ov.renderers.ovrtx_renderer_kernels` to match the
  ``warp-lang`` 1.13 API (the ``wp.math`` namespace was removed).
