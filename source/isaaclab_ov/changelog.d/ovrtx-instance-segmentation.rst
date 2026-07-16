Added
^^^^^

* Added ``idToLabels`` (instance to USD prim path) and ``idToSemantics`` (instance to semantic label)
  mappings to the OVRTX renderer's ``instance_segmentation_fast`` output, exposed through
  ``camera.data.info["instance_segmentation_fast"]``. The mappings are decoded from the
  ``StableIdSemanticIdMap``, ``StableIdMap``, and ``SemanticIdMap`` render vars and are keyed by the raw
  ``(r, g, b, a)`` color tuple when ``colorize_instance_segmentation=True`` (matching Replicator's fast
  instance-segmentation node) or by the raw instance ID otherwise.

Changed
^^^^^^^

* Changed the colorized ``semantic_segmentation`` ``idToLabels`` keys produced by the OVRTX renderer from the
  stringified ``"(r, g, b, a)"`` form to raw ``(r, g, b, a)`` tuples, matching Replicator's fast segmentation
  nodes and the Isaac RTX renderer. Index ``camera.data.info["semantic_segmentation"]["idToLabels"]`` with an
  ``(r, g, b, a)`` tuple instead of its string form.

Fixed
^^^^^

* Fixed the OVRTX renderer raising ``RuntimeError: Cannot convert Torch type torch.uint32`` when reading a
  non-colorized ID segmentation output (``semantic_segmentation``, ``instance_segmentation_fast``, or
  ``instance_id_segmentation_fast`` with the corresponding ``colorize_*`` flag set to ``False``) on Torch
  builds that expose ``torch.uint32``.
