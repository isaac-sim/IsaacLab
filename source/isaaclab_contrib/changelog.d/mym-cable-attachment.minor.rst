Added
^^^^^

* Added :class:`~isaaclab_contrib.cable.CableAttachmentCfg` to weld a
  :class:`~isaaclab_contrib.cable.CableObject` endpoint (``"head"`` or
  ``"tail"``) to a rigid body on another spawned asset via a Newton-native
  fixed joint, declared at scene-config time on
  :attr:`~isaaclab_contrib.cable.CableObjectCfg.attachments`. The anchor
  frame is specified on both sides of the joint via
  :attr:`~isaaclab_contrib.cable.CableAttachmentCfg.cable_local_pos`,
  :attr:`~isaaclab_contrib.cable.CableAttachmentCfg.cable_local_quat`
  (parent / cable-segment frame) and
  :attr:`~isaaclab_contrib.cable.CableAttachmentCfg.target_local_pos`,
  :attr:`~isaaclab_contrib.cable.CableAttachmentCfg.target_local_quat`
  (child / target-body frame), letting a baked geometric offset on the
  target asset be expressed as a body-local weld offset.
