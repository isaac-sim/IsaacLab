* Fixed ``extras["final_obs"]`` persisting into later steps without resets and into explicit :meth:`reset` and :meth:`reset_to` calls in
  manager-based, direct and multi-agent environments; it now only holds terminal observations captured in the current
  step.
* Fixed terminal observations of direct and multi-agent environments aliasing task buffers that the following
  reset could overwrite; they are now copied before the reset.
