Fixed
^^^^^

* Fixed cameras using the OVRTX renderer losing their MDL materials after an environment is torn
  down, which left surfaces such as the ground plane unshaded in the ``simple_shading_diffuse_mdl``
  and ``simple_shading_full_mdl`` outputs. Per-camera cleanup no longer releases the stage queries,
  tensor bindings and render products shared by every camera on the backend; those are released by
  :meth:`~isaaclab.renderers.BaseRenderer.close` when the simulation is torn down.
