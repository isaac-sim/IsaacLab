* Fixed actuator telemetry and history resets for worlds with different DOF layouts.
* Fixed step callbacks and sensor sites accumulating duplicates across hard resets.
* Fixed particle forces applying only to the first substep.
* Fixed a full garbage collection after every CUDA graph capture, which cost about 0.25 s per capture.
* Fixed :meth:`~isaaclab_newton.physics.NewtonCfg.copy` and ``replace`` raising on the derived ``class_type``.
