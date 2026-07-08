Added
^^^^^

* Added the Newton implementation of
  :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`, backed by
  Newton's inverse-dynamics API (``eval_inverse_dynamics``). The accessor previously
  raised ``NotImplementedError`` unconditionally; it now computes the compensation
  forces on Newton builds that provide the API (newton > 1.3) and keeps raising a
  descriptive ``NotImplementedError`` on older builds, including the currently pinned
  Newton commit.
