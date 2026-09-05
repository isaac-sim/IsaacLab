Changed
^^^^^^^

* Changed the pinned Newton version from ``1.5.1`` to ``1.6.0rc1``, the first release carrying
  `newton#4017 <https://github.com/newton-physics/newton/pull/4017>`_, which makes
  ``ArticulationView`` generic over custom frequencies. Reading a MuJoCo actuator attribute such
  as ``mujoco.actuator_trntype`` through an articulation view previously raised *"has custom
  frequency 'mujoco:actuator' which is not supported by ArticulationView"*.
* Changed ``warp-lang`` from ``1.16.0`` to ``1.17.0`` and ``mujoco``/``mujoco-warp`` from
  ``3.11.0`` to ``3.12.0``, which Newton ``1.6.0rc1`` requires.
