Deprecated
^^^^^^^^^^

* Deprecated ``ActuatorBase.compute(...)`` and ``DelayBuffer.compute(...)`` in favor of calling the object
  directly. Custom implementations should define ``__call__`` and use ``super().__call__(...)``. Existing
  ``compute`` overrides, mixins, and ``super().compute()`` chains remained supported through the transition;
  compatibility is scheduled for removal in Isaac Lab 3.2, after a full release cycle.
