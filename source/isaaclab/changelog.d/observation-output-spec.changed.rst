* Avoided camera captures and image processing during observation-manager construction for RGB,
  depth, normals, and segmentation terms. Their output specifications came from initialized camera
  buffers; observation values, stacking, and destination-buffer behavior remained unchanged.
* Added optional :attr:`~isaaclab.managers.ManagerTermBase.output_spec` for class-based observation
  terms to declare their raw output shape, dtype, and device without an initial evaluation.
  Terms without a specification retained their existing construction-time probe and reset.
