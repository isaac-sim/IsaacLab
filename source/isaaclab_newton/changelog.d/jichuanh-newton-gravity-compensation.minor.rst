Added
^^^^^

* Added the Newton implementation of
  :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`, backed by
  Newton's inverse-dynamics API (``eval_inverse_dynamics_passive``). The accessor previously
  raised ``NotImplementedError`` on the Newton backend; gravity compensation in
  task-space controllers (operational-space control, Pink IK) now works on Newton.
