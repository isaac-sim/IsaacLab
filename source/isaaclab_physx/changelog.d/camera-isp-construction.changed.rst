* Moved PPISP discovery and RTX exposure authoring to camera construction, before scene import.
  Authored PPISP camera attributes must be present before constructing the sensor; renderer initialization
  continued to create render products and the image-processing pipeline.
