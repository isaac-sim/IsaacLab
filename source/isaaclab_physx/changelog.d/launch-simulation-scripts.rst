Added
^^^^^

* Added :func:`~isaaclab_physx.renderers.isaac_rtx_renderer_utils.wait_for_stage_load` to run Kit app updates
  until the USD stage has no assets left to load.

Removed
^^^^^^^

* Removed ``isaaclab_physx.app.show_stage_in_viewport``. Preview a USD file by spawning it into a scene and
  rendering it with the visualizer selected with ``--viz``, as the ``convert_*`` tools do.
