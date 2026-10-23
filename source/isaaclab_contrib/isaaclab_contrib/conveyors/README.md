# Experimental conveyor support

This adapter stages Newton conveyor integration in `isaaclab_contrib`; its API is
experimental and is not exported from Isaac Lab core or the backend packages.

- `surface_velocity.SurfaceVelocitySpec` describes a straight or curved surface
  in the collision prim's local frame.
- `newton.SurfaceVelocity` uses Newton's upstream `ConveyorForceModel`. The
  adapter owns replicated surface/body bindings and solver lifecycle integration,
  including hard resets and registration before CUDA graph capture. It supports
  the Newton MJWarp solver; other Newton solvers are outside this adapter's scope.

Surface speeds and traction parameters come from the authored descriptions.
Full resets restart the startup ramp; episode resets clear stale forces for
only the selected worlds.

No meshes, assets, task registrations, or policy checkpoints are bundled here.
