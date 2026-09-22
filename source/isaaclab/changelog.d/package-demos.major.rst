Added
^^^^^

* Added curated showcases through ``isaaclab demo <name>`` and focused standalone programs through
  ``isaaclab example <name>``. Both catalogs are included in the released wheel and can be run with ``uvx``.
* Added the ``zoo`` demo, which animates several robot families and rigid objects in one deterministic scene.

Changed
^^^^^^^

* **Breaking:** Packaged maintained programs as importable modules instead of repository-level scripts. Use
  ``isaaclab demo list`` and ``isaaclab example list`` to discover them rather than invoking files under
  ``scripts/demos``.
* **Breaking:** Reclassified the following technical programs as examples without changing their public names:
  ``arl-robot-1``, ``bin-packing``, ``cables``, ``deformables``, ``haply-teleoperation``, ``heterogeneous-scene``,
  ``markers``, ``multi-asset``, ``procedural-terrain``, ``visual-color-randomization``, ``mpm-granular``,
  ``mpm-two-way-coupling``, ``camera``, ``contact-sensor``, ``frame-transformer``, ``imu``,
  ``multi-mesh-ray-caster``, ``multi-mesh-ray-caster-camera``, ``ppisp-camera``, ``pva``, ``ray-caster``, and
  ``tactile-sensor``. Replace direct ``scripts/demos`` invocations with ``isaaclab example <name>``.
* **Breaking:** Consolidated ``scripts/demos/sensors/newton_raycast_heightfield.py`` and
  ``scripts/demos/sensors/newton_raycast_moving_geometry.py`` as
  ``isaaclab example newton-raycast --scene {heightfield,moving-geometry}``.

Removed
^^^^^^^

* **Breaking:** Replaced the separate ``scripts/demos/arms.py``, ``bipeds.py``, ``hands.py``, ``quadcopter.py``, and
  ``quadrupeds.py`` galleries with ``isaaclab demo zoo``.
* **Breaking:** Removed the temporary ``ppisp_camera_ovrtx.py`` QA script. Use
  ``isaaclab example ppisp-camera --renderer newton_renderer`` instead.
