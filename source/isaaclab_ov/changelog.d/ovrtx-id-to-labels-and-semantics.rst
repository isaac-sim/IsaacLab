Added
^^^^^

* Added ``idToLabels``/``idToSemantics`` info dict support to the OVRTX renderer for the
  ``semantic_segmentation``, ``instance_segmentation_fast``, and ``instance_id_segmentation_fast``
  data types, via :mod:`isaaclab_ov.renderers.annotator_utils`, which decodes OVRTX's
  renderer-internal ``SemanticIdMap``, ``StableIdMap``, ``StableIdSemanticIdMap``, and
  ``InstanceMap`` AOVs.
