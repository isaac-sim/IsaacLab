* Added native NeRD transition recording and a graph-enabled learned Newton solver, shared by
  Cartpole and ANYmal C. Physical-model, physics-step and reset-history contracts are checked
  against native training checkpoints. PhysicsNeMo Robotics is an optional experimental dependency.
* NeRD consumes only applied joint efforts, captured after actuator evaluation and clipping and before
  the solver runs. Capture/deployment guards reject solver-side motor drives and external body wrenches.
  Physical identity retains fixed-root geometry, body flags and collision filtering; policy decimation
  repeats the actuator and learned physics step. Target-conditioned models are rejected.
* Added shared live Newton contact encoding for rough-terrain recording and learned deployment,
  with synchronized reset histories, explicit capacity checks and fixed contact-material identity.

* Delegated heading conversion, temporal cache/reset handling and learned Newton state writeback to
  PhysicsNeMo Robotics. ParallelNeRD deployment uses resident streaming inside the host CUDA graph.
