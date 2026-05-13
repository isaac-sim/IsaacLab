Fixed
^^^^^

* Fixed a spurious ``[Error][carb] Client passed into the framework is nullptr.``
  log emitted from :meth:`~isaaclab_newton.physics._cubric.CubricBindings.initialize`
  when the first ``tryAcquireInterfaceWithClient`` attempt returned null. The
  helper used to retry with ``clientName=None``, which Carbonite has rejected as
  invalid since 2018 — the retry only emitted a misleading error log. Removed
  the null-client retry; the existing ``acquireInterfaceWithClient`` fallback
  with the ``isaaclab.cubric`` client name still handles configurations where
  the plugin needs to be loaded on demand.
