# Extending and capturing a Newton simulation

The [main guide](newton-manager-design.md) covers the usual Isaac Lab workflow.
This page shows three ways to take more control: run independent simulations,
add your own manager behavior, and record physics into a larger CUDA graph.
Start with the example closest to your application.

## Run two independent simulations

A standalone `NewtonManager` takes a populated Newton builder, a `NewtonCfg`,
a timestep and a device. Each manager creates its own backend when reset.

This small pendulum builder is shared by the examples below. It creates a new
builder on every call, so the two simulations have independent construction data.

```python
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager


def make_builder(cfg):
    builder = cfg.solver_cfg.class_type.create_builder(physics_cfg=cfg)
    builder.begin_world()
    link = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
    )
    builder.joint_q[-1] = 0.2
    builder.add_articulation([joint])
    builder.end_world()
    return builder


left_cfg, right_cfg = NewtonCfg(), NewtonCfg()
left = NewtonManager(make_builder(left_cfg), left_cfg, dt=0.01, device="cuda:0")
right = NewtonManager(make_builder(right_cfg), right_cfg, dt=0.01, device="cuda:0")
try:
    left.reset()
    right.reset()
    for _ in range(100):
        left.step()
        right.step()
finally:
    left.close()
    right.close()
```

You can choose a different solver for each manager. Neither needs a
`SimulationContext`, and closing one leaves the other intact.

## Add custom manager behavior

Subclass `NewtonManager` when your application needs custom integration. Preserve
the base lifecycle and recreate buffer-dependent work after every hard reset.
For custom model construction, the corresponding override is `finalize_backend()`.

This manager binds a simple direct-effort controller. It uses a persistent input
buffer whose contents can change between graph replays. Use the appropriate target
fields instead when controlling a model through native actuators.

```python
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager, StepPhase


class EffortManager(NewtonManager):
    def reset(self, soft=False):
        super().reset(soft)
        if soft:
            return

        control = self.backend.control
        effort = self.effort = wp.zeros_like(control.joint_f)

        def apply_effort():
            wp.copy(control.joint_f, effort)

        self.register_step_callback(
            apply_effort, StepPhase.CONTROL, graphable=True, name="direct_effort"
        )


cfg = NewtonCfg()
manager = EffortManager(make_builder(cfg), cfg, dt=0.01, device="cuda:0")
try:
    manager.reset()
    manager.effort.fill_(0.1)
    for _ in range(100):
        manager.step()
finally:
    manager.close()
```

To use this class in Isaac Lab, select it through the configuration. This example
reuses the `build_scene` application stub from the main guide:

```python
from isaaclab.sim import SimulationCfg, SimulationContext

sim_cfg = SimulationCfg(physics=NewtonCfg(class_type=EffortManager))
sim = SimulationContext(sim_cfg)
try:
    scene = build_scene(sim)
    sim.reset()
    sim.physics_manager.effort.fill_(0.1)
    for _ in range(100):
        sim.step(render=False)
        scene.update(dt=sim_cfg.dt)
finally:
    SimulationContext.clear_instance()
```

If your application constructs the instance itself, the equivalent entry point is
`SimulationContext(sim_cfg, physics_manager=EffortManager())`. The context supplies
its configuration, device and timestep and owns the manager's lifecycle.

Most custom device work only needs a callback. Use `CONTROL` before actuators,
`STATE_FORCE` before each solver substep, or `POST_STEP` after each physics tick.
`STATE_FORCE` receives the current input state; the other callbacks take no
arguments. Native sensors refresh after `POST_STEP`. `POST_ACTUATOR` is available
for telemetry after the final physics tick's actuator evaluation.

Keep the returned callback handle if you need to unregister the work later.
Changing callbacks invalidates the schedule, so rebuild any caller-owned graph
that recorded it. For ordinary assets and sensors, `PHYSICS_READY` is the lifecycle
event used to bind callbacks to each new backend.

## Capture a larger device step

The functional API is useful when your application owns the execution loop.
`NewtonBackend` holds the live simulation data; module functions initialize and
operate on it. The example below reuses `make_builder` and records two physics
ticks into a caller-owned graph without a warmup physics step.

```python
from isaaclab_newton.physics import NewtonBackend, NewtonCfg
from isaaclab_newton.physics import newton_backend as nb

cfg = NewtonCfg()
builder = make_builder(cfg)
cfg.solver_cfg.class_type.prepare_solver_builder(builder, cfg.solver_cfg)
model = builder.finalize(device="cuda:0")
backend = NewtonBackend(model, cfg, dt=0.01)
try:
    nb.init_solver(backend)
    backend.steps_per_call = 2
    nb.prepare(backend)

    def device_step():
        # Add initialized, graph-safe device work before or after physics here.
        nb.record_step(backend)

    graph = nb.capture_graph(backend.device, device_step)
    for _ in range(100):
        graph.launch()
finally:
    backend.close()
```

`prepare()` allocates the schedule and scratch without advancing physics.
`record_step()` records physics work into the active capture. `capture_graph()`
coordinates Torch and Warp on the same stream and keeps temporary allocations
alive for replay. You can record several prepared backends in `device_step()`.

All recorded work must support capture. Initialize callback buffers first, and
keep their identities stable. The outer runner owns host time bookkeeping;
recording does not update it. Rebuild the graph after replacing buffers, changing
callbacks or changing the step count.

### Compose two simulations in one graph

Explicit backend arguments make composition straightforward. This helper accepts
two initialized managers on the same CUDA device, such as `left` and `right` from
the first example. Call it inside their `try` block, before closing either manager:

```python
from isaaclab_newton.physics import newton_backend as nb


def capture_pair(left, right):
    left.prepare()
    right.prepare()

    def advance_both():
        nb.record_step(left.backend)
        nb.record_step(right.backend)

    return nb.capture_graph(left.backend.device, advance_both)


# While both managers are initialized and alive:
# graph = capture_pair(left, right)
# graph.launch()
```

Each backend retains its own model, state and solver. The outer graph determines
the execution order. A coupled application can insert graph-safe transfers between
the two calls; this helper advances independent simulations and introduces no
physical interaction by itself. The caller maintains host time bookkeeping and
rebuilds the graph if either backend is replaced.

### Include resets when needed

For a state reset, write the selected worlds' state, call
`nb.invalidate_worlds(backend, world_mask)`, then step. The boolean device mask has
one entry per backend world and can change contents between graph replays.
Use `nb.forward(backend, force=True)` to commit a reset without advancing time.

For model-property changes, such as supported mass or friction edits, queue the
matching `newton.ModelFlags` through `nb.mark_model_changed`. The next step commits
pending property changes together. Empty masks skip the solver refresh on CUDA.
Property-changing resets inside an outer graph use `nb.capture_graph` so conditional
refresh operations retain their scratch storage.

The [solver extension guide](../../../docs/source/developer-tools/extending_newton_solvers.rst)
provides the full reset, sensor and solver-hook contracts. Follow the
[backend implementation](../isaaclab_newton/physics/newton_backend.py) when building
an integration that needs finer control over those operations.
