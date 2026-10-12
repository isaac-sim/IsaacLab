* The keyboard typing command, the lift gravity curriculum, the NIST difficulty scheduler and the deploy keypoint grasp
  rewards set ``supports_graph_capture = False``: they keep host state that changes between steps, so a captured
  environment step runs them eagerly. UR10 particle push, Franka-Pour and the G1 trocar assembly set
  ``capture_step = False``: their terms cache state by the host step counter or rebind host-held task state.
* The in-hand reorientation direct environments draw new goals with a mask instead of compacting indices and refresh
  their state snapshots in place, so their step captures with only the episode reset as an eager island. New goals
  are sampled for every environment and kept where a goal was reached.
