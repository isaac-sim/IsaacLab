* Fixed G1 symmetry augmentation on the velocity tasks: the waist yaw ``torso_joint`` and the Dex3 thumb rotation
  ``*_zero_joint`` now have explicit mirror signs, and the vector mirror factors are cached so the augmentation can run
  inside captured CUDA graphs, as SAC's updates do. The Dex3 hands are not mirror images, so finger mirroring is
  approximate.
