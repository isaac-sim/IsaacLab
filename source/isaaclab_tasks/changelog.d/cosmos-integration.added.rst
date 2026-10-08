* Added a ``cosmos`` preset to ``Isaac-Reorient-Cube-Shadow-Camera-Direct`` that
  used a resident Cosmos service to generate 640 x 640 RGB observations from depth
  controls for one environment. Playback required the run's feature-extractor
  checkpoint instead of the default pretrained fallback.
* Added capture-aligned feature-extractor supervision that retained cube-pose
  targets while generated RGB images were held and invalidated them on reset.
