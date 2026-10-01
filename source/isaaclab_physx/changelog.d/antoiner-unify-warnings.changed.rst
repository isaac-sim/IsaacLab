* Changed the contact sensor debug-visualization notice and the surface gripper stage-attribute fallbacks
  to use ``logger.warning`` instead of ``warnings.warn``, since they report runtime conditions that the
  caller cannot fix.
* Changed :class:`~isaaclab_physx.app.KitLauncher` to reuse the console logging handlers from
  ``isaaclab.app.logging_utils``, removing the stderr warning handler once Kit's log bridge is active.
