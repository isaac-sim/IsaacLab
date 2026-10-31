* Changed the SO101 IK-Abs stack task to reset the arm to its nominal joint pose without joint
  noise and to use a full-pose clutch home. Policies or datasets that relied on randomized
  initial arm joints should restore ``events.randomize_joint_state.params["std"]``.
