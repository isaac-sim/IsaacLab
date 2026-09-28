Fixed
^^^^^

* Migrated maintained Franka Reach, Drawer, Lift, and deformable tasks to the shared flat asset with
  backend-specific physics and gripper-only collisions by default. The ``arm_collisions`` preset retained
  primitive arm contacts for tasks that need them.
* Corrected Reach action, frame-transformer, and actuator contracts for the shared asset, and fixed Franka
  Lift reset sampling and success-driven motion regularization without changing Kuka-Allegro rewards. Franka
  Lift's automatic ``physics=physx`` retained the eight-shape task on Isaac Sim; select explicit
  ``physics=ovphysx`` for the cube-only kitless path. Existing Franka Lift checkpoints should be re-evaluated
  because the reset distribution changed.

Removed
^^^^^^^

* **Breaking:** Removed the unqualified ``Isaac-Reorient-Franka`` task. There is no maintained Franka
  replacement; keep an external task configuration if you rely on it, or use
  ``Isaac-Reorient-KukaAllegro`` for a supported in-hand reorientation task.
