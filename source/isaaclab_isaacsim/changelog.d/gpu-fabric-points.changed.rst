* Changed Kit RTX rendering of particle clouds and cables, such as Newton MPM particles and cable objects,
  in a GPU simulation. Kit RTX now reads their positions directly from GPU Fabric. Before, Isaac Lab copied
  the positions to the CPU at each update, and the CPU waited for the GPU each time.
