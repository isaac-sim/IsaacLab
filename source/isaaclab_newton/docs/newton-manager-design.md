# Newton manager design

The simulation has three layers: an instance-owned integration manager, an explicit
backend containing one model's runtime data, and a stateless solver adapter. The
same backend functions serve Isaac Lab and standalone applications.

## Ownership

| Component | Owns or implements |
| --- | --- |
| `NewtonManager` | Builder, backend lifetime, callbacks, views, site requests, replication metadata, control bindings and decimation. |
| `NewtonBackend` | Model, states, control, solver, contacts, collision pipeline, sensors, actuators, dirty masks, scratch buffers and step graph. |
| `NewtonSolver` adapter | Stateless construction, capability, stepping, reset and forward-kinematics hooks over an explicit backend. |
| `SimulationContext` | Isaac Lab integration and the attached manager's lifetime. |

Managers share no simulation state. Each asset, sensor and view retains its owner.
`backend.solver_adapter` selects solver behavior; `NewtonCfg.class_type` independently
selects the integration manager. Factory dispatch uses `backend_name`.

## Public API

With independently populated Newton builders:

```python
from isaaclab_newton.physics import NewtonCfg, NewtonManager

left = NewtonManager(builder_a, NewtonCfg(), dt=0.01, device="cuda:0")
right = NewtonManager(builder_b, NewtonCfg(), dt=0.01, device="cuda:0")
try:
    left.reset()
    right.reset()
    left.step()
    right.step()
finally:
    left.close()
    right.close()
```

| API | Contract |
| --- | --- |
| `reset(soft=False)` | Build and initialize without advancing physics; a soft reset retains the backend. |
| `prepare()` | Build the schedule and allocate scratch without advancing physics. |
| `step()` | Advance the configured physics steps and update host bookkeeping. |
| `record_step()` | Record device work into an outer CUDA capture, without host time updates. |
| `forward()` | Commit pending state/model changes without advancing physics. |
| `register_step_callback(fn, phase, graphable=True)` | Insert work at an explicit phase; invalidate the compiled schedule. |
| `finalize_backend()` | Public override for custom model/backend construction. |
| `close()` | Release this manager's resources and callbacks. |

For downstream integration, subclass `NewtonManager` and pass either
`NewtonCfg(class_type=MyManager)` in the simulation configuration or an instance via
`SimulationContext(sim_cfg, physics_manager=my_manager)`. The context supplies its
configuration, device and timestep; an already attached manager cannot be reused.
`SimulationContext` remains a singleton; standalone managers do not require it.

For direct control, use the functional API:

```python
from isaaclab_newton.physics import NewtonBackend
from isaaclab_newton.physics import newton_backend as nb

backend = NewtonBackend(model, NewtonCfg(), dt=0.01)
nb.init_solver(backend)
nb.register_step_callback(backend, controller, nb.StepPhase.CONTROL)
nb.prepare(backend)
graph = nb.capture_graph("cuda:0", lambda: nb.record_step(backend))
graph.launch()
# Release backend resources when finished: backend.close().
```

The model and controller buffers must already be initialized. Multiple backends
can be recorded into the same graph. New solvers subclass `NewtonSolver` and are
selected through `solver_cfg.class_type`; they need no new manager.

## Internal execution

Hard reset closes the previous backend, dispatches `MODEL_INIT`, finalizes the
model, publishes geometry, dispatches `PHYSICS_READY` to bind consumers, and then
constructs the solver and contacts. Solver construction sees the model properties
authored by consumers. Site requests survive rebuilding without duplicate injection.

`StepGraph` flattens decimation and substeps into ordered operations:

```text
each physics step:
  solver preparation → collision → CONTROL → Newton actuators
  → POST_ACTUATOR (final physics step only)
  each substep: staged forces → STATE_FORCE → solver
  → normalize output into stable state_0 → POST_STEP → native sensors
after decimation: normalize actuator history → clear forces
```

Configured collision refreshes also run between substeps. Ordinary action terms,
controllers and scene command writers bind to `CONTROL` in their existing order.
The environment omits their duplicate host execution. Native contact and IMU
updates run every physics step, keeping feedback current within decimation.

Consecutive graphable operations form captured segments. A fully graphable
schedule becomes one captured simulation step; eager operations retain their exact
position. `record_step` rejects unsupported solvers or eager operations before
executing work. Action terms and actuators explicitly declare
`supports_graph_capture`; Python history or host-dependent behavior stays eager.

The first ordinary `step` executes eagerly, then captures without another physics
advance. `prepare` plus `record_step` supports outer capture without a warmup step.
`capture_graph` coordinates Torch and Warp on one stream and retains both allocation
lifetimes, including conditional-node scratch; replay uses `graph.launch()`.

## Resets and performance

State writes use `invalidate_worlds`/`invalidate_fk` followed by `forward` or `step`.
Property writes call `mark_model_changed(backend, flags, env_mask)`; the next boundary
combines pending categories into one solver refresh. CUDA conditional execution
skips empty masks without host synchronization. Property-changing resets inside
an outer graph require `capture_graph` to retain their scratch storage.

Stable buffers, cached schedules, lazy scratch allocation and one initialization
pass reduce startup work and per-step dispatch. Single-state solvers avoid redundant
state buffers. Captured derived-state reads recompute from device state; wrench
and reset selection use device buffers rather than captured Python flags.
Finite-difference accelerations retain scene-publication cadence.

See the [solver extension guide](../../../docs/source/developer-tools/extending_newton_solvers.rst)
for hook contracts, coupling and migration details, and the
[functional implementation](../isaaclab_newton/physics/newton_backend.py) for scheduling
and capture internals.
