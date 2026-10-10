* Fixed height-field terrain generation raising a shape-mismatch ``ValueError`` for sizes, horizontal scales and
  border widths whose inner pixel count lands just below an integer in floating point, such as a 10 m terrain with a
  0.1 m horizontal scale and a 0.4 m border.
