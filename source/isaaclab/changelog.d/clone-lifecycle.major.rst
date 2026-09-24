Changed
^^^^^^^

* **Breaking:** Required an explicit clone plan before starting or resetting a simulation.
  ``InteractiveScene`` continued to own its clone lifecycle. Standalone callers must declare
  their assets with ``clone_plan_from_env_0`` or publish a ``make_clone_plan`` with exact
  authored roots, then call ``replicate(plan)`` before ``reset()`` or ``play()``.
  Empty simulations must explicitly publish and replicate an empty plan.
