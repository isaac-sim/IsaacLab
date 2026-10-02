* Added GPU Fabric particle-point updates on Kit 110.4 and newer when RTX Points geometry
  streaming is disabled. Older Kit versions, streamed Points, and curve geometry retain CPU
  Fabric destinations because Kit 110.4's streamed Points do not render GPU position changes.
