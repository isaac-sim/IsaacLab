* Fixed multi-minute stalls on Windows in Newton CUDA graph capture after the renderer's first frame.
  Python array allocations made during capture now use ordinary device memory held until the graph is
  released, instead of graph memory, on all platforms.
