Added
^^^^^

* Added :func:`~isaaclab.sim.utils.has_deformable_body_api` to detect applied deformable body API
  schemas, including schemas authored as unregistered tokens (e.g. by Newton).

Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.sim.schemas.define_deformable_body_properties` to no longer
  remove a pre-existing deformable body setup before authoring a new one. Clear any previous setup
  before calling it, or use :func:`~isaaclab.sim.schemas.modify_deformable_body_properties` to update
  properties on an existing deformable body.

Fixed
^^^^^

* Fixed Newton kitless deformable mesh spawning to avoid requiring Kit-only PhysX helpers.
* Fixed detection of deformable body API schemas authored as unregistered tokens (e.g. by Newton),
  so such assets are modified in place instead of being re-defined during spawning.
