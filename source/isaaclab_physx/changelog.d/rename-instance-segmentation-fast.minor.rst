Changed
^^^^^^^

* **Breaking:** Updated :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` to use the renamed
  ``"instance_segmentation"`` data type (previously ``"instance_segmentation_fast"``).
  The renderer maps this key to the Replicator ``"instance_segmentation_fast"`` annotator internally.
