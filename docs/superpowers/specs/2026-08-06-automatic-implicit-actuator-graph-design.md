# Automatic Implicit Actuator CUDA Graph Design

## Context

PR 6839 executes exact built-in `ImplicitActuator` groups through a fused Warp
kernel. Compatible groups are aggregated before execution, and every command,
state, parameter, output, joint-index, and compatibility buffer consumed by the
kernel is allocated and bound during `ActuatorCollection` construction. On CUDA,
the existing `_WarpLaunchCache` records a pointer-stable `wp.Launch`, but normal
articulation execution still submits that launch eagerly on every step.

The accepted 4096-environment PhysX Franka benchmark used this cached-eager path.
Nsight found no CUDA graph replay, so the benchmarked performance improvement did
not include graph-launch amortization.

## Goal

Automatically capture and replay the complete Isaac Lab implicit-actuator compute
sequence when one articulation contains only exact built-in `ImplicitActuator`
execution batches on CUDA. The optimization must require no public configuration,
must compose with caller-owned CUDA capture, and must preserve an eager fallback.

## Non-goals

- Capturing PhysX, OVPhysX, or Newton command submission.
- Capturing the physics solver or the complete environment step.
- Capturing mixed implicit/explicit collections.
- Capturing `ImplicitActuator` subclasses or third-party actuator models.
- Capturing Ideal PD, DC motor, neural, delayed, remotized, or other stateful
  actuator execution.
- Introducing a public graph option or backend-specific execution strategy.
- Porting the global actuator collection's general execution scheduler, prefix
  graphs, graph leases, compatibility epilogues, or dedicated stream handoff.

## Eligibility

Eligibility is resolved once after execution batches are built. A collection is
eligible only when all of the following hold:

- Its execution device is CUDA.
- It has at least one execution batch.
- Every execution batch's exact actuator type is `ImplicitActuator`.
- Backend-native actuator execution is inactive.
- The private class constant `_ENABLE_IMPLICIT_CUDA_GRAPH` is enabled. This
  switch exists only for focused tests and same-revision benchmarks; it is not
  public API.

CPU collections, empty collections, mixed collections, custom subclasses, and
backend-native collections continue through their existing paths without
attempting capture.

## Graph ownership and execution

One `ActuatorCollection` owns at most one implicit execution graph. Capturing the
whole eligible collection preserves execution-batch order and gives a single
graph launch even if construction produced more than one implicit batch. Normal
aggregation is expected to reduce common configurations such as Franka to one
batch and one fused kernel.

The collection stores private graph state:

- `_use_implicit_cuda_graph`, resolved once at construction;
- `_implicit_cuda_graph`, containing the captured `wp.Graph` or `None`; and
- `_implicit_cuda_graph_capture_failed`, recording whether the single local
  capture attempt failed.

`ActuatorCollection.compute()` retains the backend-native precheck. For an
eligible Lab-owned collection it then routes all implicit batches through one
private execution method:

1. If Warp reports capture on the collection's current dispatch stream through
   `wp.get_device(self.device).is_capturing`, execute the existing batch launches
   directly. `_WarpLaunchCache` already emits ordinary `wp.launch` calls during
   capture, so the caller records the kernels into its graph. A nested local
   graph is neither created nor replayed. Capture begun externally on a different
   stream is outside this contract because Warp does not expose it through the
   collection's current dispatch state.
2. If a local graph exists, replay it on the current stream.
3. If the previous local capture attempt failed, execute the existing cached
   eager sequence.
4. Otherwise, lazily capture every implicit batch in order using
   `wp.ScopedCapture` on the collection device, retain the resulting graph, and
   replay it for the current compute call.

The first automatic capture occurs during ordinary warm-up execution. Capture
does not include `submit_commands()`, solver writes, physics stepping, command
setters, parameter writes, or resets.

## Pointer and lifecycle contract

The graph relies on the same construction-time pointer-stability contract as the
existing recorded `wp.Launch` objects:

- collection command, processed-command, torque, and compatibility buffers are
  allocated once;
- batch joint indices and Warp argument lists are built once;
- aggregate actuator parameters and outputs retain stable storage;
- supported runtime gain writers update existing tensors in place; and
- backend state views retain stable storage for the articulation lifetime.

Actuator membership is already immutable after construction. Articulation
reinitialization creates a new collection, so a graph is never rebound to a new
generation. The graph remains private and lives no longer than its collection.
Directly replacing actuator tensors is outside the existing cached-launch
contract and is not made a supported operation by this change.

## Failure behavior

Local graph capture is attempted at most once per eligible collection. If capture
raises an exception, the collection records the failure, emits one warning with
the reason, computes the current outputs eagerly, and remains cached-eager for
the rest of its lifetime. A capture failure never aborts simulation and is never
silently reported as graph execution.

A Warp-visible capture on the current dispatch stream never mutates local graph
state. If such a capture begins after a local graph was created, the collection
still emits the underlying Warp kernels into the outer graph instead of nesting
the existing graph replay.

## Verification

Development follows test-driven development. CUDA-focused tests must fail on the
pre-change implementation and then prove:

- the first eligible compute captures and replays one collection graph;
- subsequent computes replay that graph and consume changed command and state
  contents without changing pointers;
- graph outputs match cached-eager outputs exactly;
- a Warp-visible capture on the current dispatch stream records the underlying
  implicit sequence and leaves local graph state untouched, including when a
  local graph already exists;
- capture failure produces correct eager outputs, warns once, and is not retried;
- CPU, mixed, custom-subclass, and backend-native collections do not attempt
  automatic capture; and
- existing aggregation, gain-write, command-routing, and backend tests remain
  unchanged in behavior.

The focused tests are followed by the relevant PhysX, Newton, and OVPhysX
articulation tests, documentation generation when public documentation changes,
and all repository pre-commit hooks.

## Benchmark and decision

Benchmarking compares eager and automatic execution from the same source
revision. A hashed external launcher sets
`ActuatorCollection._ENABLE_IMPLICIT_CUDA_GRAPH` to `False` for eager or `True`
for graph before importing and invoking the benchmark entry point. The launcher
then calls `isaaclab.benchmark.entrypoints.runtime.run()` with the remaining
arguments. This selects the construction path before the environment is created,
adds no public configuration surface, and does not change the captured execution
body. The launcher source and SHA-256 are retained with every attempt.

First, an actuator-only CUDA microbenchmark measures the exact built-in implicit
collection at 4096 environments. It reports host submission latency and
synchronized elapsed time for cached eager and graph replay, verifies exact
outputs, and confirms that no allocation or recapture occurs during measured
iterations.

Second, the runtime entry point at
`source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py` runs
`Isaac-Reach-Franka` with regular PhysX. Each member is equivalent to:

```text
./isaaclab.sh -p <hashed-launcher.py> --mode <eager|graph> \
  --task Isaac-Reach-Franka --num_envs 4096 \
  --warmup_steps 100 --num_steps 1000 --seed 42 \
  --visualizer none --benchmark_formatter schema \
  --output_path <unique-member-directory> physics=isaacsim_physx
```

The command intentionally omits `--measure_sync_step`. The primary timing
boundary is therefore the harness's `host_return` environment-step measurement,
matching the accepted historical Franka protocol: `EnvironmentStepTimingRecorder`
records wall time around each complete `env.step()` call after the 100 excluded
steps. The actuator-only microbenchmark supplies the synchronized measurement.

Eager and automatic modes run in fresh processes as twelve counterbalanced
pairs: six eager-then-graph and six graph-then-eager. The primary result is the
paired graph/eager environment-step latency ratio, accompanied by per-mode
median latency, throughput, dispersion, and a 5000-resample paired bootstrap
95% confidence interval using seed 42.

An attempt is accepted only when both members exit successfully; produce exactly
one schema bundle; report the frozen source SHA, launcher hash, dependency
versions, hardware, task, backend, environment count, seed, warm-up count, and
measurement count; contain finite positive timing samples; begin below 5% GPU
utilization with no competing compute process; report no active throttling; and
finish with member endpoint temperatures within 5 degrees Celsius. A rejected
pair and every reason remain in the evidence record rather than being silently
replaced.

A separate untimed smoke launcher retains the constructed collection long enough
to assert that graph mode owns a non-`None` `_implicit_cuda_graph`, that capture
did not fall back, and that eager mode owns no graph. One graph-mode profile uses
100 warm-up and 200 measured steps under `nsys profile --trace=cuda,nvtx,osrt`;
the exported CUDA API and kernel tables must contain graph launch evidence.
Benchmark results are not labeled as graph results if either validation fails.
The evidence is presented without a predetermined minimum-win threshold. The
implementation is retained or removed after reviewing the measured gain, noise,
memory cost, and code complexity with the maintainer.

## Success criteria

- Eligible implicit actuator computation automatically replays one CUDA graph.
- Caller-owned capture records the same kernels without nested graph ownership.
- No public API, configuration, backend behavior, or CPU behavior changes.
- Capture failure degrades once and safely to the existing cached-eager path.
- Eager and graph outputs remain exact for changing inputs.
- The Franka A/B evidence is reproducible and clearly separates graph impact
  from the previously measured aggregation improvement.
