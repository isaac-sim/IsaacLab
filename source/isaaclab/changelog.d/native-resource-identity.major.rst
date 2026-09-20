Changed
^^^^^^^

* **Breaking:** Simplified native resource registration to ``sim.get_or_create_backend(backend_cfg)``.
  Move constructor inputs into a ``BackendCfg`` whose ``class_type`` constructs the resource from cfg;
  equal configurations of the same concrete type shared one resource. Registered cfgs were retained
  without copying; finalize them before registration and treat them as read-only. Release a resource
  with ``sim.clear_backend(backend_cfg)``. Resources must implement ``clear()``.
* **Breaking:** Separated clone contexts from native ownership. Replace clone-context registration
  through ``get_or_create_backend(Context, ...)`` with ``sim.clone_contexts[Context] = Context(...)``.
* Added ``field(metadata={"copy": False})`` support to ``configclass`` for borrowed native inputs.
  Construction, ``copy()``, and ``replace()`` preserved these references without changing the
  independent copying of ordinary configuration fields.
