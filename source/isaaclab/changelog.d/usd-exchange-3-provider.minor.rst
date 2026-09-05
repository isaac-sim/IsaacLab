Changed
^^^^^^^

* **Breaking:** Changed the pinned standalone OpenUSD provider from ``usd-exchange`` 2.3.0 to
  3.0.0. In kit-less installs ``pxr`` is now OpenUSD **26.08** instead of 25.05. Environments
  that pin Isaac Lab against OpenUSD 25.05 outside Kit must re-resolve, and any code compiled
  or generated against the 25.05 ``pxr`` ABI must be rebuilt against 26.08. Kit-backed runs are
  unaffected: Kit serves its own OpenUSD from its extension roots. The URDF and MJCF importer
  packages need no change, since each requires only ``usd-exchange>=2.2.2``.
