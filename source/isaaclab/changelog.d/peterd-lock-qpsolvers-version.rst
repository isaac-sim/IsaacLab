Fixed
^^^^^

* Pinned ``qpsolvers==4.11.0`` to keep Pink IK working. ``qpsolvers`` 4.12.0
  dropped the ``primal_start`` kwarg, causing ``pin-pink==3.1.0`` to raise
  ``TypeError`` inside ``solve_ik``;
  :class:`~isaaclab.controllers.pink_ik.PinkIKController` then silently
  fell back to returning the current joints, making IK a no-op.
