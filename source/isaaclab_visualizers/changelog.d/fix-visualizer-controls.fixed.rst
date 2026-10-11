* Fixed Kit toolbar controls so they reflected the running state, paused and resumed training while keeping the app
  responsive, and did not start PhysX when another physics backend owned the simulation.
* Fixed Newton GL camera keyboard controls when ImGui captured only mouse input and kept the viewer compatible with
  Newton's updated fullscreen image handling.
* Prevented unchanged Rerun instance appearance from being retransmitted every frame during training.
