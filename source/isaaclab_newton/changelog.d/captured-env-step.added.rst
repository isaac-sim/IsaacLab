* Newton contact, IMU, PVA, joint-wrench, frame-transformer and ray-cast sensors declare ``supports_graph_capture``.
  Articulations report ``reset_supports_graph_capture`` as False when a Newton actuator is not graphable, so a
  captured environment step resets them eagerly.
