* Changed ``set_term_cfg`` of the Warp reward, event, termination and command managers to record the manager's
  stages again only when the term's function, parameters or capturability change, or an event term's trigger
  settings. Setting an unchanged configuration, as curricula do on every reset, keeps the recorded stages.
* Changed the Warp reward manager to check zero term weights inside the recorded reward stage, so a weight set
  to or from zero applies without recording the stage again. This needs CUDA 12.4 or newer; with older CUDA
  versions a weight crossing zero records the stage again.
