Supported Features
==================

PhysX is the broadest backend in Isaac Lab. It is the reference for behaviour
parity and supports every public asset, sensor, and renderer surface in the
framework. Tasks built before Isaac Lab 3.0 ran on PhysX, and the bulk of the
shipped tasks still default to the PhysX preset.

The summary below is intentionally coarse; consult each component's API
documentation for fine-grained capability details.

Core Simulation
---------------

* Articulation API (multi-link articulations, fixed-base and floating-base
  articulations, single-body articulations modeled as rigid bodies)
* Rigid Object and Rigid Object Collection APIs
* Soft-body and particle simulation (legacy — not exposed through the
  Isaac Lab asset surface but available through PhysX schemas)
* CPU and GPU pipelines; GPU is the default for the vectorized RL workloads


Sensors
-------

* Contact Sensor
* IMU
* Frame Transformer
* Ray Caster
* PVA
* Joint Wrench Sensor
* Visuo-tactile sensor
* Camera (RTX) — see :doc:`../../sensors/camera`


Rendering
---------

* RTX renderer (real-time and path-traced)
* Tiled rendering for vectorized RGB / depth / segmentation


Tasks and Workflows
-------------------

* Direct and Manager-based workflows
* All ``isaaclab_tasks`` environments default to the PhysX preset unless the
  task explicitly opts in to a different backend
* Imitation learning and motion-generation pipelines (Mimic, motion generators)


Known Caveats
-------------

* PhysX requires Isaac Sim and Omniverse Kit to be installed.
* GPU buffer sizes are static and must be tuned per task — see
  :doc:`configuration`.
* ``enable_stabilization`` can corrupt contact-force readings reported through
  the contact sensor; disable it if you rely on the contact sensor for
  force-magnitude observations.
* The PhysX TGS solver behaviour can differ from Newton's MJWarp solver on
  stiff contact stacks; if you are porting a task to Newton, expect to retune
  contact-handling parameters.
