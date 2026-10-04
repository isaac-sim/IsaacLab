* Allowed ``--video`` with ``--frontend warp``; it previously raised a :class:`ValueError`.
  Without ``--viz``, the Warp frontend records from an auto-created headless Newton GL visualizer
  instead of Kit, which needs a full Isaac Sim install. The torch frontend still gets Kit.
