Fixed
^^^^^

* Fixed simulation mesh discovery in :class:`~isaaclab_contrib.deformable.DeformableObject` to detect
  deformable sim API schemas authored as unregistered tokens (e.g. by Newton), so surface deformables
  no longer fall back to treating the visual mesh as the simulation mesh.
