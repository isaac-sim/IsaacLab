* Removed duplicated PPISP configuration discovery from renderer initialization. Cameras now resolved
  ``isp_cfg`` during construction, before scene import; renderer initialization continued to create the
  image-processing pipeline from the prepared configuration.
* Consolidated lens-distortion checks into render-data creation and read segmentation labels directly
  from the simulation-owned stage.
