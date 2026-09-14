Added
^^^^^

* Added ``IsaacContrib-PickPlace-Locomanipulation-Deformable-G1-Abs``, a G1 locomanipulation
  pick-and-place task whose graspable object is a deformable cube rather than a rigid one. The
  task runs only on Newton: MJWarp has no deformable support, so the scene is partitioned between
  a ``rigid`` MJWarp entry owning the G1 articulation and a ``soft`` VBD entry owning the object's
  particles, joined by :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` with the hand links
  exposed as proxies. It follows the maintained ``Isaac-Lift-Soft-Franka`` task, and needs the
  ``tetrahedralization`` extra for the volume mesh.

  Adapting that recipe from a fixed-base arm to a walking humanoid needed four changes:

  * Coupled solvers reject contact sensors, since contact forces live in per-entry buffers. The
    per-hand sensors and the controller haptics they drive are dropped.
  * ``CouplerEntryCfg.include_static_shapes`` takes every static shape into one entry, which gave
    the ground plane to VBD and left MJWarp with no ground: the robot's feet sank to 18 cm below
    the floor. The ground and the tabletop are now assigned by shape label, so the robot keeps its
    ground contact and the object still rests on the table.
  * Full-surface rigid-soft contact samples each rigid shape's signed-distance field, and the G1's
    hand colliders are meshes without one. Contact falls back to per-vertex.
  * A soft body spawned intersecting the table is ejected rather than pushed out as a rigid body
    would be, so the object starts clear of the tabletop rather than at the rigid task's height.

  The object has no single rigid pose, so the orientation-dependent observation terms are removed;
  position-based terms read the mean of the nodal positions.
