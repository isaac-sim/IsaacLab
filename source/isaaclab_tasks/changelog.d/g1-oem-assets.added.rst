* Added an opt-in ``presets=oem`` selection for G1 29-DOF rough locomotion, its depth student, motion tracking,
  the three pretrained G1 Terrain courses, locomanipulation and its Mimic workflows, fixed-base upper-body IK,
  Inspire pick-and-place and both trocar assembly tasks. Install the ``simready-retrieve`` CLI to use this selection.
* The OEM locomotion assets use the sole of Unitree's 29-DOF model, and rough locomotion keeps only the torso, head,
  pelvis and foot colliders. Under ``presets=oem``, rough locomotion and motion tracking use the Unitree URDF ankle
  and waist roll/pitch limits, and rough locomotion also terminates on head contact.
