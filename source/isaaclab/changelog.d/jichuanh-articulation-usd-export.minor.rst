Added
^^^^^

* Added selected-environment USD export after initialization and before startup events. Preserved source physics, resources and defaults, supplemented fixed configuration overrides, and omitted live state samples without clearing source velocities.

Fixed
^^^^^

* Avoided exporting actuator values already supplied by the imported asset, including equivalent values from other USD schemas.
* Updated standalone USD Exchange to 3.0.0 and removed single-threaded payload parsing workarounds.
