Added
^^^^^

* Added the Newton implementation of
  :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`, backed by
  Newton's inverse-dynamics API (``eval_inverse_dynamics``). The accessor previously
  raised ``NotImplementedError`` on the Newton backend; gravity compensation in
  task-space controllers (operational-space control, Pink IK) now works on Newton.

Changed
^^^^^^^

* Changed the pinned Newton commit to ``b24dc255392126fe42a609ea6c4e4c8dca8009cc``,
  which includes the inverse-dynamics feature (joint-space mass matrix, gravity and
  Coriolis compensation forces). The MuJoCo dependencies move from the 3.8 to the
  3.10 series with it, overriding Isaac Sim's ``mujoco-warp`` pin.
