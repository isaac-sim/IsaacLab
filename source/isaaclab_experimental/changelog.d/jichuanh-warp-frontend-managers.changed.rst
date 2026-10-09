* Changed ``set_term_cfg`` of the Warp reward, event, termination and command managers to record the manager's
  stages again only when the term's function, parameters or capturability change, an event term's trigger
  settings change, or a reward term's weight turns zero or nonzero. Setting an unchanged configuration, as
  curricula do on every reset, keeps the recorded stages, and other weight changes apply to the recorded reward
  stage, which reads the weights on the device.
