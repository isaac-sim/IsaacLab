Fixed
^^^^^

* Fixed a spurious ``[Error][carb] Client passed into the framework is nullptr.``
  log emitted from :meth:`~isaaclab.cloner._fabric_notices.FabricNoticeBindings.initialize`
  when an environment imports IsaacLab outside Kit (e.g. remote asset resolution
  via ``omni.client``). The helper was passing ``clientName=None`` as a fallback
  to ``tryAcquireInterfaceWithClient``; Carbonite has rejected null client names
  since 2018, so the call only emitted a misleading error log and never returned
  a valid interface. The fallback has been removed; the helper still fails closed
  when Fabric is unavailable, with no impact on the cloning speedup when Fabric
  is present.
