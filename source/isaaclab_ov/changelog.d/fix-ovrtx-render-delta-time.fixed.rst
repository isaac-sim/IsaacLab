* Fixed the OVRTX renderer stepping temporal effects such as motion blur and motion vectors with a fixed
  1/60 s delta, by passing the simulation's rendering time step to each render submission.
