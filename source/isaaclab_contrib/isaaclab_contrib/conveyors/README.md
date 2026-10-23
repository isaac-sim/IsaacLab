# Experimental conveyor support

This adapter stages Newton conveyor integration in `isaaclab_contrib`; its API is
experimental and is not exported from Isaac Lab core or the backend packages.

- `surface_velocity.SurfaceVelocitySpec` describes a straight or curved surface
  in the collision prim's local frame.
- `newton.SurfaceVelocity` uses Newton's upstream `ConveyorForceModel`. The
  adapter owns replicated surface/body bindings and solver lifecycle integration,
  including hard resets and registration before CUDA graph capture. It supports
  the Newton MJWarp solver; other Newton solvers are outside this adapter's scope.

The adapter retains commanded speeds while disabled, restarts its startup ramp
on a full reset, and clears stale forces for selected worlds. The surface spec
controls traction parameters.

No meshes, assets, task registrations, or policy checkpoints are bundled here.

The optional `physx.SurfaceVelocity` adapter uses native `PhysxSurfaceVelocityAPI`
on CPU. Author its kinematic collision surfaces with `apply_surface_velocity_api`
before simulation starts, then bind the live adapter after PhysX parses the scene.
It requires Isaac Sim and uses authored contact materials for traction.
