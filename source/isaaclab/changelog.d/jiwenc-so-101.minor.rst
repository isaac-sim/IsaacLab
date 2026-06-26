Added
^^^^^

* Added an ``"adaptive_dls"`` ``ik_method`` to :class:`~isaaclab.controllers.DifferentialIKController`:
  a manipulability-aware damped least squares whose damping ramps from ``lambda_min`` toward
  ``lambda_max`` as the smallest task-Jacobian singular value drops below ``sigma_thresh``
  (Maciejewski-Klein style), keeping low-DOF / near-singular arms well-conditioned.
* Added an optional per-axis :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.orientation_weight`
  (scalar or ``(wx, wy, wz)``) that soft-weights the orientation rows of a ``"pose"`` task, so an
  arm that cannot serve a full 6-DOF pose degrades gracefully instead of leaking orientation error
  into position.
* Added null-space joint-limit avoidance to :class:`~isaaclab.controllers.DifferentialIKController`
  via :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.joint_limit_avoidance_gain` /
  :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.joint_limit_avoidance_margin` and
  :meth:`~isaaclab.controllers.DifferentialIKController.set_joint_pos_limits`. When enabled, a
  center-seeking bias is projected into the null space of the position rows so it never perturbs
  the commanded end-effector position; :class:`~isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction`
  injects the joint limits automatically.

Changed
^^^^^^^

* Changed :meth:`~isaaclab.controllers.DifferentialIKController.set_command` to renormalize the
  commanded quaternion for absolute ``"pose"`` commands, hardening the controller against slightly
  non-unit quaternion inputs. Existing unit-quaternion callers are unaffected.
