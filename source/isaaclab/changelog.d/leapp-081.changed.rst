* **Breaking:** Updated the optional LEAPP integration from 0.7.2 to 0.8.1. New bundles used
  quantity kinds and per-dimension ``axes`` instead of ``element_names``. Isaac Lab deployment
  retained support for reading legacy bundles.
* **Breaking:** Replaced ``element_names`` and ``element_names_resolver`` in
  ``leapp_tensor_semantics`` and ``LeappTensorSemantics`` with ``axes`` and ``axes_resolver``.
  Command configurations used ``axes`` instead of ``element_names``. Replaced component-label
  constants with ``*_COMPONENT_AXIS`` descriptors and body/joint resolvers with axis resolvers.
* Labeled environment dimensions as ``AxisKind.SYSTEM`` while preserving tensor shapes,
  explicit user layouts, element ordering, and raw recurrent feedback.
* Standardized velocity and twist components as ``x``, ``y``, ``z``, ``angular_x``, ``angular_y``,
  and ``angular_z``. Used released quantity helpers for built-in inputs, commands, and PD gains,
  including planar twist setpoints with the subset ``x``, ``y``, and ``angular_z``.
* Added command reference and expression-frame metadata, contact-duration kinds, cable and
  joint-wrench axis labels, and constant metadata for the deprecated default root state.
  Used built-in quantities for cable pose/velocity and world-frame ray-caster pose inputs.
* Reduced export adapters to session-scoped RGB normalization and tensor-broadcasting patches.
  Traced original action processing and destination writes while retaining explicit recurrent,
  history, and EMA feedback state, including the clone needed to preserve EMA state inputs.
