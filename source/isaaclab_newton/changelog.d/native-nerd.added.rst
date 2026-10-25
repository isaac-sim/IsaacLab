* Added native NeRD transition recording and a graph-enabled learned Newton solver, shared by
  Cartpole and ANYmal C. Physical-model, control-period and reset-history contracts are checked
  against native training checkpoints. PhysicsNeMo Robotics is an optional experimental dependency.
* Shared capture/deployment guards reject omitted implicit position drives and external inputs.
  Physical identity retains fixed-root geometry, body flags and collision filtering; effort-conditioned
  physics supports policy decimation with the explicit actuator kept outside NeRD.
