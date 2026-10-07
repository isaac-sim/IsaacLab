* Fixed Windows hangs from lazy MuJoCo-Warp GPU allocations during CUDA graph capture by running
  the first requested physics step eagerly before recording the graph for subsequent steps.
