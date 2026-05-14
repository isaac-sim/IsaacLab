Changed
^^^^^^^

* Bumped the ``newton[sim]`` pin from ``v1.2.0rc2`` to ``v1.2.0rc3``
  across :mod:`isaaclab_newton`, :mod:`isaaclab_physx` (``[newton]``
  extra), :mod:`isaaclab_visualizers` (3×), and
  ``tools/wheel_builder/res/python_packages.toml``. Restores the
  pin briefly active on develop between `isaac-sim/IsaacLab#5024
  <https://github.com/isaac-sim/IsaacLab/pull/5024>`_ and
  `isaac-sim/IsaacLab#5566
  <https://github.com/isaac-sim/IsaacLab/pull/5566>`_, this time
  keeping the canonical ``newton[sim] @ git+...`` form with the
  ``[sim]`` extra preserved everywhere.
* No IsaacLab-side ``mujoco`` / ``mujoco-warp`` pin change — the
  transitive ``mjwarp`` bump flows in through ``newton[sim]`` since
  `isaac-sim/IsaacLab#5566
  <https://github.com/isaac-sim/IsaacLab/pull/5566>`_ dropped the
  explicit pins.
