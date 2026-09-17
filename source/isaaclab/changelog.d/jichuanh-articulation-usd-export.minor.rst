Added
^^^^^

* Added selected-environment USD export after initialization and before startup events. Preserved source physics, resources and defaults, supplemented fixed configuration overrides, and omitted live state samples without clearing source velocities.

Fixed
^^^^^

* Avoided exporting actuator values already supplied by the imported asset, including equivalent values from other USD schemas. Declared actuator configuration bindings on data-property USD decorators and discovered them during initialization, eliminating the parallel configuration-to-data table. Preserved existing dictionary resolution and backend-specific drive semantics.

* Preserved existing scene content when copied dependencies needed an export-resource namespace already occupied by source prims.
