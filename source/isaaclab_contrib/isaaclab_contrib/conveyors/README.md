# Experimental conveyor support

These adapters stage conveyor integration in `isaaclab_contrib`; their API is
experimental and is not exported from Isaac Lab core or the backend packages.

- `surface_velocity.SurfaceVelocitySpec` describes a straight or curved surface
  in the collision prim's local frame. `SurfaceVelocityView` describes the shared
  control operations.
- `newton.SurfaceVelocity` uses Newton's upstream `ConveyorForceModel`. The
  adapter owns replicated surface/body bindings and solver lifecycle integration,
  including hard resets and registration before CUDA graph capture. It requires
  the Newton backend.
- `physx.SurfaceVelocity` uses native `PhysxSurfaceVelocityAPI` on CPU.
  Author the API before simulation parsing with `apply_surface_velocity_api`.
  This path requires the PhysX backend and Kit; GPU dynamics are unsupported.

Both adapters retain commanded speeds while disabled and clear selected worlds'
runtime state on reset. The Newton adapter supports live traction controls;
PhysX uses authored contact materials and rejects unsupported traction setters.

No meshes, assets, task registrations, or policy checkpoints are bundled here.
