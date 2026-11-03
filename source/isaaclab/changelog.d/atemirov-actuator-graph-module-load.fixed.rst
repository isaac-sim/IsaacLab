* Fixed the first reset with PhysX physics taking many minutes on a machine without cached GPU kernels. Capturing the
  Newton actuator graphs forced Warp to load and compile every imported module; it now loads only the kernels it
  records.
