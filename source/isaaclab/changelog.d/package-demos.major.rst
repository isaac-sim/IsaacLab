Added
^^^^^

* Added curated showcases through ``isaaclab demo <name>`` and focused standalone programs through
  ``isaaclab example <name>``. Both catalogs are included in the released wheel and can be run with ``uvx``.
* Added the ``zoo`` demo, which animates several robot families and rigid objects in one deterministic scene.
* Added a Newton GL program selector that puts curated demos ahead of focused examples.
* Reused the Isaac Lab terminal startup screen for packaged demos and examples.
* Added :meth:`~isaaclab.sim.SimulationContext.add_reset_callback` and
  :meth:`~isaaclab.sim.SimulationContext.remove_reset_callback` for callbacks that must be registered before a
  script creates its simulation context.

Changed
^^^^^^^

* **Breaking:** Moved maintained programs from ``scripts/demos`` into the repository-level ``examples`` directory,
  with showcases in ``examples/demos`` and focused programs in topic directories such as ``examples/mpm``. Use
  ``isaaclab demo <name>`` or ``isaaclab example <name>`` instead of direct script paths.
* **Breaking:** Reclassified the following technical programs as examples without changing their public names:
  ``arl-robot-1``, ``bin-packing``, ``cables``, ``deformables``, ``haply-teleoperation``, ``heterogeneous-scene``,
  ``markers``, ``multi-asset``, ``newton-dominoes``, ``procedural-terrain``, ``visual-color-randomization``,
  ``mpm-granular``, ``mpm-two-way-coupling``, ``camera``, ``contact-sensor``, ``frame-transformer``, ``imu``,
  ``multi-mesh-ray-caster``, ``multi-mesh-ray-caster-camera``, ``ppisp-camera``, ``pva``, ``ray-caster``, and
  ``tactile-sensor``. Replace direct ``scripts/demos`` invocations with ``isaaclab example <name>``.
* Changed ``isaaclab demo h1-locomotion`` to run on Newton with the Newton GL viewer by default. Robots are driven
  with I/J/K/L, selected with N, and followed with C. Pass ``--physics isaacsim_physx`` for PhysX.
* **Breaking:** Consolidated ``scripts/demos/sensors/newton_raycast_heightfield.py`` and
  ``scripts/demos/sensors/newton_raycast_moving_geometry.py`` as
  ``isaaclab example newton-raycast --scene {heightfield,moving-geometry}``.

Removed
^^^^^^^

* **Breaking:** Replaced the separate ``scripts/demos/arms.py``, ``bipeds.py``, ``hands.py``, ``quadcopter.py``, and
  ``quadrupeds.py`` galleries with ``isaaclab demo zoo``.
* **Breaking:** Removed the temporary ``ppisp_camera_ovrtx.py`` QA script. Use
  ``isaaclab example ppisp-camera --renderer newton_renderer`` instead.
