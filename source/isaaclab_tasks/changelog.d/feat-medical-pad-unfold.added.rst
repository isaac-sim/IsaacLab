* Added ``Isaac-Cosmos3-Medical-Pad-Unfold-v0``, an H2 medical-pad unfolding task whose joint state and camera frames
  come from the Cosmos3 world model of ``isaaclab_neural`` instead of a physics engine. It uses joint-position
  actions, joint and image observations, resets from recorded start states and a classifier-progress reward
  and success and failure terminations, judged frame by frame inside each generated chunk and reported when the chunk
  completes.
