Changed
^^^^^^^

* Bumped the ``newton[sim]`` pin from ``v1.2.0rc2`` to ``v1.2.0``
  (stable) across :mod:`isaaclab_newton`, :mod:`isaaclab_physx`
  (``[newton]`` extra), :mod:`isaaclab_visualizers` (3×), and
  ``tools/wheel_builder/res/python_packages.toml``. Upstream release
  notes: `newton-physics/newton v1.2.0
  <https://github.com/newton-physics/newton/releases/tag/v1.2.0>`_.
* No IsaacLab-side ``mujoco`` / ``mujoco-warp`` pin change — the
  transitive ``mjwarp`` bump flows in through ``newton[sim]`` since
  `isaac-sim/IsaacLab#5566
  <https://github.com/isaac-sim/IsaacLab/pull/5566>`_ dropped the
  explicit pins.
