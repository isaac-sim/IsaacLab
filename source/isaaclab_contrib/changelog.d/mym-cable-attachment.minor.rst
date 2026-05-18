Added
^^^^^

* Added :class:`~isaaclab_contrib.cable.CableAttachmentCfg` to weld a
  :class:`~isaaclab_contrib.cable.CableObject` endpoint (``"head"`` or
  ``"tail"``) to a rigid body on another spawned asset via a Newton-native
  fixed joint, declared at scene-config time on
  :attr:`~isaaclab_contrib.cable.CableObjectCfg.attachments`.
