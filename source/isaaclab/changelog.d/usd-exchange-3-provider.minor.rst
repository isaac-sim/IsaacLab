Changed
^^^^^^^

* Changed the pinned standalone OpenUSD provider from ``usd-exchange`` 2.3.0 to 3.0.0. In
  kit-less installs ``pxr`` is now OpenUSD 26.08 instead of 25.05. The URDF and MJCF importer
  packages resolve against it unchanged, since each requires only ``usd-exchange>=2.2.2``.
