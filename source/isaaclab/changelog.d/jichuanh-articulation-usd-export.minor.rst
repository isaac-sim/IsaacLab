Added
^^^^^

* Added selected-environment USD export after initialization and before startup events. Preserved source physics, resources and defaults, supplemented fixed configuration overrides, and omitted live state samples without clearing source velocities.

Fixed
^^^^^

* Avoided exporting actuator values already supplied by the imported asset, including equivalent values from other USD schemas. Shared actuator property bindings between initialization and override provenance instead of maintaining a separate export mapping.
