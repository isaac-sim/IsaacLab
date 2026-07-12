Changed
^^^^^^^

* Changed :class:`~isaaclab.cloner.ReplicateSession` to publish the clone plan when
  the session enters instead of when it exits, so entities constructed inside the
  session (sensors, frame views, ray casters) resolve prims through the plan
  consistently with post-replication code. A failed session unpublishes its plan.
* Changed frame views to a uniform two-phase lifecycle: every backend view can be
  constructed alongside the rest of the scene and completes its initialization
  through its backend's own lifecycle (Newton registers frame sites at construction
  so they clone with the scene; PhysX and OVPhysX resolve prims once physics is
  ready). The camera sensor constructs its frame view at construction on every
  backend.

Fixed
^^^^^

* Fixed weak-reference physics callbacks raising ``ReferenceError`` during event
  dispatch after their owner was garbage-collected; the registry entry is now
  deregistered when the owner is collected.
