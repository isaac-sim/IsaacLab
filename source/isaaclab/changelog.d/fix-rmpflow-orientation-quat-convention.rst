Fixed
^^^^^

* Fixed a quaternion convention bug in :class:`~isaaclab.controllers.rmp_flow.RmpFlowController`
  where the end-effector orientation target was re-converted to ``(x, y, z, w)`` even though
  IsaacLab quaternions are already in that order. The spurious conversion scrambled the target
  orientation handed to RMPFlow, causing the arm to drift away from its commanded pose (e.g. the
  Agibot RMPFlow place tasks no longer hold their reset pose under a zero relative command).
