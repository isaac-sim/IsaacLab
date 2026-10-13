* Fixed lazy MJCF and URDF conversion returning an older USD after the importer wrote a changed asset to a numbered
  folder. The cache now records the generated file and reuses only the most recent matching conversion. Old
  hash-only cache records cause one conversion because they do not identify the generated file.
