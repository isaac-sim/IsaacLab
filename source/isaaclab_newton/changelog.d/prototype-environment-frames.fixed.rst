* Captured the authored source environment origin when importing deformables so expanded visual geometry metadata
  remained correctly placed when the authoring stage contained untransformed prototypes.
* Used authored source frames instead of planned destination poses when composing Newton worlds, keeping bodies,
  cameras, and deformables aligned when prototype environments were unplaced or remapped.
