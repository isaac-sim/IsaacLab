Fixed
^^^^^

* Fixed a broken ``pxr`` (OpenUSD) runtime on x86_64 caused by co-installing
  ``usd-core`` and ``usd-exchange``. Both wheels vendor a complete ``pxr``
  package at different USD versions, so installing both left a mixed
  installation that raised ``RuntimeError: extension class wrapper for base
  class ... Tf_PyEnumWrapper has not been created yet`` on ``import pxr``.
  ``usd-exchange`` is now installed only on ``aarch64``/``arm64`` (where
  ``usd-core`` has no wheel); x86_64 uses ``usd-core`` as the sole ``pxr``
  provider.
