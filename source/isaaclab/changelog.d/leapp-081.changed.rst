* **Breaking:** Updated the optional LEAPP integration from 0.7.2 to 0.8.1. New bundles used
  quantity kinds and per-dimension ``axes`` instead of ``element_names``. Isaac Lab deployment
  retained support for reading legacy bundles.
* Retained ``element_names``, ``element_names_resolver``, and the list-returning label helpers.
  Added optional ``axes`` and ``axes_resolver`` for detailed layouts, taking precedence over simple
  labels. Converted simple labels to LEAPP axes during export without requiring LEAPP objects in
  command configurations. Accepted legacy velocity component spellings for built-in velocity kinds.
* Labeled environment dimensions as ``AxisKind.SYSTEM`` while preserving tensor shapes,
  explicit user layouts, element ordering, and raw recurrent feedback.
* Standardized velocity and twist components as ``x``, ``y``, ``z``, ``angular_x``, ``angular_y``,
  and ``angular_z``. Used released quantity helpers for built-in inputs, commands, and PD gains,
  including planar twist setpoints with the subset ``x``, ``y``, and ``angular_z``.
* Added contact-duration kinds, cable and joint-wrench axis labels, and constant metadata
  for the deprecated default root state.
  Used built-in quantities for cable pose/velocity and world-frame ray-caster pose inputs.
* Reduced export adapters to session-scoped RGB normalization and tensor-broadcasting patches.
  Traced original action processing and destination writes while retaining explicit recurrent,
  history, and EMA feedback state, including the clone needed to preserve EMA state inputs.
