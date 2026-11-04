* **Breaking:** Migrated LEAPP exports to version 0.8.0 quantity kinds, dimension axes, setpoint flags,
  and body-frame spatial metadata. External bundle consumers must read ``axes`` instead of
  ``element_names``; Isaac Lab deployment retained support for legacy bundles.
* Added command reference and expression-frame metadata and session-scoped function patches
  for RGB normalization and broadcast quaternion rotation. Removed the export-only action
  clone now that in-place action writes are traced automatically.
* Kept OSC task-space gains internal to exported effort computation instead of labeling them
  as joint gains. Resolved command frames independently of public body ordering and left
  yaw-only planar references unspecified.
* Removed the RGB export observation proxy by tracing destination writes directly in the
  normalization function patch.
* Removed redundant export-only clones from EMA state registration and processed-action fallback outputs.
* **Breaking:** Replaced decorator ``element_names`` and ``element_names_resolver`` with
  explicit LEAPP ``axes`` and ``axes_resolver`` declarations. Replaced component-label constants
  with ``*_COMPONENT_AXIS`` descriptors and body/joint resolvers with axis resolvers. Command
  configurations now declare ``axes`` instead of ``element_names``. Removed kind-based axis
  inference and corrected ray-caster coordinates to use component axes.
