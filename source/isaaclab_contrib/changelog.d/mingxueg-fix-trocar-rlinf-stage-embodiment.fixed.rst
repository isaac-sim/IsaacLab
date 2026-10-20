* Fixed the RLinf extension ignoring ``embodiment_tag_id`` for tags that RLinf already registers. RLinf maps
  ``new_embodiment`` to projector 10, so GR00T N1.5 checkpoints fine-tuned with ``new_embodiment`` = 31 decoded
  actions with the wrong projector. A configured ID now overrides the registered one.
