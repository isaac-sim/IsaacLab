Fixed
^^^^^

* Fixed demonstration replay stepping once after all episodes completed.
* Fixed :meth:`~isaaclab.controllers.DifferentialIKController.set_command` handling of
  unnormalizable absolute-pose quaternions, which produced a NaN target orientation. Such
  commands now hold the current end-effector orientation, or identity when none is provided.
