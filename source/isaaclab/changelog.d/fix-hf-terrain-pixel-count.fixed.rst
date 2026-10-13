* Fixed height-field terrains dropping a pixel or a vertical step when a length is a multiple of its scale but does
  not divide exactly in floating point, such as a 9.7 m size, a 0.3 m border or step width, or a 0.145 m step
  height. This also fixed the shape-mismatch ``ValueError`` raised for a 10 m terrain with a 0.1 m horizontal scale
  and a 0.4 m border. Terrains generated from such values changed; clear the terrain cache to regenerate them.
