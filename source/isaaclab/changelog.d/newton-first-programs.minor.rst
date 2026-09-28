Changed
^^^^^^^

* Changed ``markers``, ``procedural-terrain``, ``camera``, ``frame-transformer``, ``imu``,
  ``multi-mesh-ray-caster-camera``, ``pva``, and ``ray-caster`` examples to start with Newton and Newton GL by default.
  Pass ``--physics isaacsim_physx --viz kit`` to retain the previous PhysX/Kit launch.
* Changed ``bin-packing`` to start with Newton VBD and Newton GL by default while retaining heterogeneous bins.
  Pass ``--physics isaacsim_physx --viz kit`` to use its previous PhysX/Kit launch.
* Changed ``arl-robot-1`` to start with Newton MJWarp and Newton GL by default, using the shared articulation and
  wrench APIs for its thruster forces. Pass ``--physics isaacsim_physx --viz kit`` to use its previous PhysX/Kit launch.
* Changed the ``pick-and-place`` demo default to one environment. Pass ``--num_envs 32`` for the previous layout.

Fixed
^^^^^

* Fixed source-checkout program discovery when a stale, incomplete installed-program directory exists.
