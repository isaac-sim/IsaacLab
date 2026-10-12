* Preserved the original ``ModuleNotFoundError`` when Viser or its dependencies were missing,
  removing the installation-hint wrapper.
* Simplified Viser initialization and camera updates around its native client API, preserving deferred
  camera poses until clients connected and removing unused metadata and repeated capability checks.
