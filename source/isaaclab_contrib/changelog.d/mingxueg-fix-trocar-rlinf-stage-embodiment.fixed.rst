* Fixed the RLinf extension ignoring ``embodiment_tag_id`` for tags that RLinf already registers. RLinf maps
  ``new_embodiment`` to projector 10, so GR00T N1.5 checkpoints fine-tuned with ``new_embodiment`` = 31 decoded
  actions with the wrong projector. A configured ID now overrides the registered one.
* Fixed multi-GPU RLinf training crashing the renderer at startup. Each env worker now launches Isaac Sim with
  ``multi_gpu=False``, so its renderer only uses the worker's GPU instead of every GPU on the node.
