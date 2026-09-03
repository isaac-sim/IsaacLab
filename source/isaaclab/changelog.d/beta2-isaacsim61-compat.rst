Fixed
^^^^^

* Fixed URDF and MJCF conversions producing assets without active physics when importers emitted an
  unselected ``Physics`` variant set.
* Fixed GUI startup with Isaac Sim 6.1 by loading the PhysX Fabric extension required by the PhysX
  manager.
* Fixed source-install failures caused by overlapping standalone OpenUSD providers by using
  ``usd-exchange`` consistently.
* Kept Fabric change-notice suspension a no-op when ``usdrt`` is unavailable outside a live Kit
  application.
