* Fixed Warp environments replaying CUDA graphs that read freed simulation buffers after the simulation was
  reset or stopped. The environments now record their stages again on the next step.
