Fixed
^^^^^

* Fixed a SIGSEGV crash when calling ``SimulationContext.play()`` a second time after
  ``SimulationContext.stop()`` without an intervening ``reset()``, e.g. registering a raw
  ``omni.timeline`` event subscription and cycling play/stop twice on the GPU PhysX pipeline.
  ``PhysxManager`` now detaches the PhysX stage on ``stop()`` so the next play's automatic
  re-warmup reattaches cleanly instead of calling ``attach_stage`` on a stage that PhysX still
  considered attached, which corrupted its internal view registry.
