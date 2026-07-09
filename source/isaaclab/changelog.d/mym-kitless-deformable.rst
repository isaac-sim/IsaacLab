Fixed
^^^^^

* Fixed Newton kitless deformable mesh spawning to avoid requiring Kit-only PhysX helpers.
* Fixed detection of deformable body API schemas authored as unregistered tokens (e.g. by Newton)
  via the new :func:`~isaaclab.sim.utils.has_deformable_body_api` helper, so such assets are
  modified in place instead of being re-defined during spawning.
