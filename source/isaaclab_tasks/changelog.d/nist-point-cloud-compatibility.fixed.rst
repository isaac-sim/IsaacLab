* Added explicit point-sampling seed, assembly sampling order, and observation-only robot USD settings
  to the NIST variant task. These settings allowed checkpoints to regenerate their observation points
  after catalog reorderings or physical robot asset updates, without loading saved point clouds.
  Existing defaults were unchanged; playback of older checkpoints required matching their original
  sampling order and observation geometry through the perception observation term parameters.
