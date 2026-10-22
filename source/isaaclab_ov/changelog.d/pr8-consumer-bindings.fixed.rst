* Fixed multi-mesh ray-cast targets spanning several rigid bodies and binding to bodies
  in other environments. Sensors and targets now read shared body poses by exact path,
  preserving environment order across native clone batches and articulation links.
