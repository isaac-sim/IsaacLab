Fixed
^^^^^

* Fixed ``scripts/tutorials/01_assets/add_new_robot.py`` failing at the Jetbot
  velocity-target setter when the action was a CPU ``torch.Tensor``. The
  tutorial now allocates the wheel-velocity templates on ``sim.device`` and
  uses :meth:`~isaaclab.assets.BaseArticulation.set_joint_velocity_target_index`.
