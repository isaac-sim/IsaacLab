Added
^^^^^

* Added a contributed manager-based environment with guarded, counter-rotating force-driven racetrack
  conveyors, robust primitive and closed-mesh belt colliders, a MuJoCo Menagerie Franka, and an interactive
  Newton-viewer cube-goal selector.
* Added task-local, schema-aligned conveyor descriptions and a tensorized control view while retaining a single,
  kitless Newton force owner with CUDA-graph and hard-reset-safe lifecycle binding.
* Added the opt-in ``IsaacContrib-Conveyor-Franka-PhysX-CPU-v0`` reference task, which explicitly
  rejects GPU dynamics because the supported native surface-velocity path can drop conveyor contacts.

Changed
^^^^^^^

* Allowed :func:`isaaclab_tasks.utils.parse_env_cfg` callers to preserve a task's configured simulation device by
  passing ``device=None``.
* Kept action-rate penalties finite for rejected NaN or infinite policy commands by tracking the sanitized
  commands accepted by the task's action terms.
