* Fixed :func:`~isaaclab.utils.configclass` failing to construct a subclass whose base class defines a read-only
  property. The post-init copy only skipped properties declared on the instance's own class, so it tried to assign
  an inherited property and raised :class:`AttributeError`.
