* Fixed ``extras["final_obs"]`` missing when a reset requested from the viewer ends episodes on a step without an
  automatic reset, in manager-based, direct and multi-agent environments with ``compute_final_obs`` enabled.
